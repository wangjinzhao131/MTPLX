"""GPU arithmetic/shape checks for the opt-in small-batch GEMV experiment."""

import unittest

import mlx.core as mx

from mtplx.local_optimizations.qwen27 import verify_kernel as candidate


class VerifyKernelTests(unittest.TestCase):
    def setUp(self):
        mx.random.seed(619)

    def arrays(self, bits, group, k=512, n=16):
        weight = mx.random.normal((n, k)).astype(mx.float16)
        arrays = mx.quantize(weight, bits=bits, group_size=group)
        # Compiling unevaluated random/quantization graphs changes the fixture.
        mx.eval(*arrays)
        return arrays

    def assert_exact(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        mx.eval(actual, expected)
        delta = actual.astype(mx.float32) - expected.astype(mx.float32)
        self.assertTrue(mx.array_equal(actual, expected).item(), {
            "unequal_values": mx.sum(actual != expected).item(),
            "max_abs_error": mx.max(mx.abs(delta)).item(),
        })

    def variants(self, bits, m):
        return [name for name in candidate.VARIANTS
                if (name != "paired4" or (bits, m) == (4, 2))
                and (name != "paired8" or (bits, m) == (8, 2))
                and (name != "quartet4" or (bits, m) == (4, 2))
                and (name != "triplet4" or (bits, m) == (4, 3))]

    def test_native_fast_arithmetic_all_vector_counts_and_production_k(self):
        for bits, group in ((4, 32), (8, 64)):
            for k in (5120, 17408, 6144):
                arrays = self.arrays(bits, group, k=k)
                for m in (1, 2, 3, 4):
                    x = mx.random.normal((m, k)).astype(mx.float16)
                    mx.eval(x)
                    expected = mx.quantized_matmul(
                        x, *arrays, transpose=True, bits=bits, group_size=group)
                    for variant in self.variants(bits, m):
                        with self.subTest(bits=bits, k=k, m=m, variant=variant):
                            self.assert_exact(candidate.qmv(
                                x, *arrays, bits=bits, group_size=group,
                                variant=variant), expected)

    def test_strided_inputs_leading_dims_and_compiled(self):
        for bits, group in ((4, 32), (8, 64)):
            full = self.arrays(bits, group, n=32)
            arrays = tuple(a[::2] for a in full)
            x = mx.random.normal((1, 6, 1024)).astype(mx.float16)[:, ::2, ::2]
            mx.eval(x, *arrays)
            expected = mx.quantized_matmul(
                x, *arrays, transpose=True, bits=bits, group_size=group)
            for variant in self.variants(bits, 3):
                def fn(value):
                    return candidate.qmv(value, *arrays, bits=bits,
                                         group_size=group, variant=variant)
                with self.subTest(bits=bits, variant=variant):
                    self.assert_exact(fn(x), expected)
                    self.assert_exact(mx.compile(fn)(x), expected)

    def test_zero_and_constant_operands(self):
        for bits, group in ((4, 32), (8, 64)):
            arrays = mx.quantize(mx.full((16, 512), 0.5, mx.float16),
                                 group_size=group, bits=bits)
            mx.eval(*arrays)
            for value in (0, -0.5, 0.5):
                x = mx.full((4, 512), value, mx.float16)
                expected = mx.quantized_matmul(
                    x, *arrays, transpose=True, bits=bits, group_size=group)
                for variant in self.variants(bits, 4):
                    self.assert_exact(candidate.qmv(
                        x, *arrays, bits=bits, group_size=group, variant=variant), expected)

    def test_q4_private_packed_words_high_and_low_halves(self):
        # Every nibble is distinct; zero-scale constant weights hid the bug.
        packed = mx.array([0x76543210, 0xFEDCBA98], dtype=mx.uint32)
        weight = mx.tile(packed, (16, 32))
        scales = mx.full((16, 16), 0.5, dtype=mx.float16)
        biases = mx.full((16, 16), -3, dtype=mx.float16)
        for start in (0, 16, 496):
            positions = mx.array([start, start + 7, start + 8, start + 15])
            x = (mx.arange(512)[None, :] == positions[:, None]).astype(mx.float16)
            expected = mx.broadcast_to(
                (positions % 16 * 0.5 - 3).astype(mx.float16)[:, None], (4, 16))
            for variant in self.variants(4, 4):
                with self.subTest(start=start, variant=variant):
                    self.assert_exact(candidate.qmv(
                        x, weight, scales, biases, bits=4, group_size=32,
                        variant=variant), expected)

    def test_paired_strided_inputs_compiled_replay_and_fused_rows(self):
        arrays = tuple(a[::2] for a in self.arrays(4, 32, n=32))

        def paired(x):
            return candidate.qmv(x, *arrays, bits=4, group_size=32,
                                 variant="paired4")

        compiled = mx.compile(paired)
        for _ in range(2):
            x = mx.random.normal((1, 4, 1024)).astype(mx.float16)[:, ::2, ::2]
            mx.eval(x, *arrays)
            expected = mx.quantized_matmul(x, *arrays, transpose=True,
                                           bits=4, group_size=32)
            self.assert_exact(paired(x), expected)
            self.assert_exact(compiled(x), expected)
            self.assert_exact(paired(x), candidate.qmv(
                x, *arrays, bits=4, group_size=32, variant="stream4"))
        parts = [self.arrays(4, 32, n=n) for n in (8, 16)]
        packed = tuple(mx.concatenate([p[i] for p in parts], axis=0) for i in range(3))
        mx.eval(*packed)
        expected = mx.concatenate([mx.quantized_matmul(
            x, *part, transpose=True, bits=4, group_size=32) for part in parts], axis=-1)
        self.assert_exact(candidate.qmv(x, *packed, bits=4, group_size=32,
                                       variant="paired4"), expected)

    def test_paired_nibbles_cross_k_blocks_and_constant_inputs(self):
        packed = mx.array([0x76543210, 0xFEDCBA98], dtype=mx.uint32)
        weight = mx.tile(packed, (16, 64))
        scales = mx.full((16, 32), 0.5, dtype=mx.float16)
        biases = mx.full((16, 32), -3, dtype=mx.float16)
        mx.eval(weight, scales, biases)
        for start in (0, 16, 496, 512, 1008):
            for offsets in ((0, 7), (8, 15)):
                positions = mx.array([start + offset for offset in offsets])
                amplitudes = mx.array([1, -0.5], dtype=mx.float16)
                x = ((mx.arange(1024)[None, :] == positions[:, None]).astype(mx.float16)
                     * amplitudes[:, None])
                expected = mx.broadcast_to(
                    ((positions % 16 * 0.5 - 3).astype(mx.float16)
                     * amplitudes)[:, None], (2, 16))
                with self.subTest(start=start, offsets=offsets):
                    self.assert_exact(candidate.qmv(
                        x, weight, scales, biases, bits=4, group_size=32,
                        variant="paired4"), expected)
        for value in (0, -0.5, 0.5):
            x = mx.full((2, 1024), value, dtype=mx.float16)
            expected = mx.quantized_matmul(x, weight, scales, biases,
                                           transpose=True, bits=4, group_size=32)
            self.assert_exact(candidate.qmv(x, weight, scales, biases,
                                           bits=4, group_size=32, variant="paired4"), expected)

    def test_paired_rejects_other_batches_and_q8(self):
        for bits, group, m in ((4, 32, 1), (4, 32, 3), (4, 32, 4), (8, 64, 2)):
            arrays = self.arrays(bits, group)
            x = mx.zeros((m, 512), dtype=mx.float16)
            with self.subTest(bits=bits, m=m), self.assertRaisesRegex(
                    ValueError, "paired4 requires M=2"):
                candidate.qmv(x, *arrays, bits=bits, group_size=group, variant="paired4")

    def test_quartet_strides_compiled_replay_and_final_row_group(self):
        # N=24 exercises a final group beyond a power-of-two row count.
        arrays = tuple(a[::2] for a in self.arrays(4, 32, k=1024, n=48))

        def quartet(x):
            return candidate.qmv(x, *arrays, bits=4, group_size=32,
                                 variant="quartet4")

        compiled = mx.compile(quartet)
        for _ in range(2):
            x = mx.random.normal((1, 4, 2048)).astype(mx.float16)[:, ::2, ::2]
            mx.eval(x, *arrays)
            expected = mx.quantized_matmul(x, *arrays, transpose=True,
                                           bits=4, group_size=32)
            self.assert_exact(quartet(x), expected)
            self.assert_exact(compiled(x), expected)
            self.assert_exact(quartet(x), candidate.qmv(
                x, *arrays, bits=4, group_size=32, variant="paired4"))
        parts = [self.arrays(4, 32, n=n) for n in (8, 16)]
        packed = tuple(mx.concatenate([p[i] for p in parts], axis=0) for i in range(3))
        x = mx.random.normal((1, 2, 512)).astype(mx.float16)
        mx.eval(x, *packed)
        expected = mx.concatenate([mx.quantized_matmul(
            x, *part, transpose=True, bits=4, group_size=32) for part in parts], axis=-1)
        self.assert_exact(candidate.qmv(x, *packed, bits=4, group_size=32,
                                       variant="quartet4"), expected)

    def test_quartet_nibbles_k_blocks_and_native_half_bias_sum(self):
        packed = mx.array([0x76543210, 0xFEDCBA98], dtype=mx.uint32)
        weight = mx.tile(packed, (16, 64))
        scales = mx.full((16, 32), 0.5, dtype=mx.float16)
        biases = mx.full((16, 32), -3, dtype=mx.float16)
        mx.eval(weight, scales, biases)
        for start in (0, 16, 496, 512, 1008):
            for offsets in ((0, 7), (8, 15), (3, 4), (11, 12)):
                positions = mx.array([start + offset for offset in offsets])
                amplitudes = mx.array([1, -0.5], dtype=mx.float16)
                x = ((mx.arange(1024)[None, :] == positions[:, None]).astype(mx.float16)
                     * amplitudes[:, None])
                expected = mx.broadcast_to(
                    ((positions % 16 * 0.5 - 3).astype(mx.float16)
                     * amplitudes)[:, None], (2, 16))
                with self.subTest(start=start, offsets=offsets):
                    self.assert_exact(candidate.qmv(
                        x, weight, scales, biases, bits=4, group_size=32,
                        variant="quartet4"), expected)
        # Cancellation makes native half quartet additions significant. Keep
        # native qmv as the reference, including its expression promotions.
        fixtures = [mx.full((2, 1024), value, mx.float16) for value in (0, -0.5, 0.5)]
        fixtures.append(mx.concatenate([mx.array(
            [[2048, 1, -2048, 0], [-2048, -1, 2048, 0]], mx.float16),
            mx.zeros((2, 1020), mx.float16)], axis=1))
        zero_weight = mx.zeros_like(weight)
        zero_scales = mx.zeros_like(scales)
        for arrays in ((weight, scales, biases), (zero_weight, zero_scales, biases)):
            mx.eval(*arrays)
            for x in fixtures:
                mx.eval(x)
                expected = mx.quantized_matmul(x, *arrays, transpose=True,
                                               bits=4, group_size=32)
                self.assert_exact(candidate.qmv(x, *arrays, bits=4, group_size=32,
                                               variant="quartet4"), expected)
                self.assert_exact(candidate.qmv(x, *arrays, bits=4, group_size=32,
                                               variant="paired4"), expected)

    def test_quartet_rejects_unsupported_vectors_quantization_and_tails(self):
        for bits, group, m in ((4, 32, 1), (4, 32, 3), (4, 32, 4), (8, 64, 2)):
            arrays = self.arrays(bits, group)
            x = mx.zeros((m, 512), dtype=mx.float16)
            with self.subTest(bits=bits, m=m), self.assertRaisesRegex(
                    ValueError, "quartet4 requires M=2"):
                candidate.qmv(x, *arrays, bits=bits, group_size=group, variant="quartet4")
        for k, n in ((768, 16), (512, 20)):
            arrays = self.arrays(4, 32, k=k, n=n)
            x = mx.zeros((2, k), dtype=mx.float16)
            with self.subTest(k=k, n=n), self.assertRaisesRegex(ValueError, "native-fast aligned K"):
                candidate.qmv(x, *arrays, bits=4, group_size=32, variant="quartet4")

    def test_triplet_nibbles_k_blocks_and_native_half_bias_sum(self):
        packed = mx.array([0x76543210, 0xFEDCBA98], dtype=mx.uint32)
        weight = mx.tile(packed, (24, 64))
        scales = mx.full((24, 32), 0.5, dtype=mx.float16)
        biases = mx.full((24, 32), -3, dtype=mx.float16)
        mx.eval(weight, scales, biases)
        for start in (0, 16, 496, 512, 1008):
            for offsets in ((0, 7, 8), (15, 3, 4), (11, 12, 1)):
                positions = mx.array([start + offset for offset in offsets])
                amplitudes = mx.array([1, -0.5, 0.25], dtype=mx.float16)
                x = ((mx.arange(1024)[None, :] == positions[:, None]).astype(mx.float16)
                     * amplitudes[:, None])
                expected = mx.broadcast_to(
                    ((positions % 16 * 0.5 - 3).astype(mx.float16)
                     * amplitudes)[:, None], (3, 24))
                with self.subTest(start=start, offsets=offsets):
                    self.assert_exact(candidate.qmv(
                        x, weight, scales, biases, bits=4, group_size=32,
                        variant="triplet4"), expected)
        fixtures = [mx.full((3, 1024), value, mx.float16) for value in (0, -0.5, 0.5)]
        fixtures.append(mx.concatenate([mx.array(
            [[2048, 1, -2048, 0], [-2048, -1, 2048, 0], [2048, -2048, 1, 0]], mx.float16),
            mx.zeros((3, 1020), mx.float16)], axis=1))
        for arrays in ((weight, scales, biases),
                       (mx.zeros_like(weight), mx.zeros_like(scales), biases)):
            mx.eval(*arrays)
            for x in fixtures:
                mx.eval(x)
                expected = mx.quantized_matmul(x, *arrays, transpose=True,
                                               bits=4, group_size=32)
                self.assert_exact(candidate.qmv(x, *arrays, bits=4, group_size=32,
                                               variant="triplet4"), expected)

    def test_triplet_rejects_other_vectors_quantization_and_tails(self):
        for bits, group, m in ((4, 32, 1), (4, 32, 2), (4, 32, 4), (8, 64, 3)):
            arrays = self.arrays(bits, group)
            x = mx.zeros((m, 512), dtype=mx.float16)
            with self.subTest(bits=bits, m=m), self.assertRaisesRegex(
                    ValueError, "triplet4 requires M=3"):
                candidate.qmv(x, *arrays, bits=bits, group_size=group, variant="triplet4")
        for k, n in ((768, 16), (512, 20)):
            arrays = self.arrays(4, 32, k=k, n=n)
            x = mx.zeros((3, k), dtype=mx.float16)
            with self.subTest(k=k, n=n), self.assertRaisesRegex(ValueError, "native-fast aligned K"):
                candidate.qmv(x, *arrays, bits=4, group_size=32, variant="triplet4")

    def test_paired8_strides_compiled_replay_and_packed_rows(self):
        arrays = tuple(a[::2] for a in self.arrays(8, 64, n=32))

        def paired(x):
            return candidate.qmv(x, *arrays, bits=8, group_size=64,
                                 variant="paired8")

        compiled = mx.compile(paired)
        for _ in range(2):
            x = mx.random.normal((1, 4, 1024)).astype(mx.float16)[:, ::2, ::2]
            mx.eval(x, *arrays)
            expected = mx.quantized_matmul(x, *arrays, transpose=True,
                                           bits=8, group_size=64)
            self.assert_exact(paired(x), expected)
            self.assert_exact(compiled(x), expected)
            self.assert_exact(paired(x), candidate.qmv(
                x, *arrays, bits=8, group_size=64, variant="shared"))
        parts = [self.arrays(8, 64, n=n) for n in (8, 16)]
        packed = tuple(mx.concatenate([p[i] for p in parts], axis=0) for i in range(3))
        mx.eval(*packed)
        expected = mx.concatenate([mx.quantized_matmul(
            x, *part, transpose=True, bits=8, group_size=64) for part in parts], axis=-1)
        self.assert_exact(candidate.qmv(x, *packed, bits=8, group_size=64,
                                       variant="paired8"), expected)

    def test_paired8_high_low_bytes_k_blocks_and_constants(self):
        # Distinct bytes cover both uint32 words and values with the sign bit set.
        packed = mx.array([0x7F400100, 0xFFFEC880], dtype=mx.uint32)
        weight = mx.tile(packed, (16, 128))
        scales = mx.full((16, 16), 0.25, dtype=mx.float16)
        biases = mx.full((16, 16), -31, dtype=mx.float16)
        byte_values = mx.array([0, 1, 64, 127, 128, 200, 254, 255], dtype=mx.float16)
        mx.eval(weight, scales, biases, byte_values)
        for start in (0, 8, 248, 256, 504, 512, 1016):
            for offsets in ((0, 3), (4, 7), (1, 6), (2, 5)):
                positions = mx.array([start + offset for offset in offsets])
                amplitudes = mx.array([1, -0.5], dtype=mx.float16)
                x = ((mx.arange(1024)[None, :] == positions[:, None]).astype(mx.float16)
                     * amplitudes[:, None])
                expected = mx.broadcast_to(
                    ((byte_values[positions % 8] * 0.25 - 31)
                     * amplitudes)[:, None], (2, 16))
                with self.subTest(start=start, offsets=offsets):
                    self.assert_exact(candidate.qmv(
                        x, weight, scales, biases, bits=8, group_size=64,
                        variant="paired8"), expected)
        for value in (0, -0.5, 0.5):
            x = mx.full((2, 1024), value, dtype=mx.float16)
            expected = mx.quantized_matmul(x, weight, scales, biases,
                                           transpose=True, bits=8, group_size=64)
            self.assert_exact(candidate.qmv(x, weight, scales, biases,
                                           bits=8, group_size=64, variant="paired8"), expected)

    def test_paired8_rejects_other_batches_and_q4(self):
        for bits, group, m in ((8, 64, 1), (8, 64, 3), (8, 64, 4), (4, 32, 2)):
            arrays = self.arrays(bits, group)
            x = mx.zeros((m, 512), dtype=mx.float16)
            with self.subTest(bits=bits, m=m), self.assertRaisesRegex(
                    ValueError, "paired8 requires M=2"):
                candidate.qmv(x, *arrays, bits=bits, group_size=group, variant="paired8")

    def test_reject_unsupported_inputs(self):
        arrays = self.arrays(4, 32)
        x = mx.zeros((2, 512), dtype=mx.float16)
        for kwargs in ({"variant": "wide"}, {"bits": 8}, {"group_size": 64}):
            options = {"bits": 4, "group_size": 32, **kwargs}
            with self.subTest(options=options), self.assertRaises(ValueError):
                candidate.qmv(x, *arrays, **options)
        for invalid_x in (x.astype(mx.float32), x[0],
                          mx.zeros((5, 512), dtype=mx.float16), x[:, :256],
                          mx.zeros((0, 512), dtype=mx.float16)):
            with self.assertRaises(ValueError):
                candidate.qmv(invalid_x, *arrays, bits=4, group_size=32)
        for invalid_arrays in ((arrays[0][:15], arrays[1][:15], arrays[2][:15]),
                               (arrays[0], arrays[1][:, :-1], arrays[2]),
                               (arrays[0].astype(mx.int32), arrays[1], arrays[2])):
            with self.assertRaises(ValueError):
                candidate.qmv(x, *invalid_arrays, bits=4, group_size=32)

    def test_partitioned_real_projection_shapes(self):
        for bits, group, widths in ((4, 32, (17408, 17408)),
                                    (8, 64, (17408, 17408)),
                                    (4, 32, (10240, 6144, 48, 48)),
                                    (4, 32, (12288, 1024, 1024))):
            parts = [self.arrays(bits, group, k=5120, n=width) for width in widths]
            packed = tuple(mx.concatenate([part[i] for part in parts], axis=0) for i in range(3))
            mx.eval(*packed)
            for m in (2, 3):
                variant = "shared" if bits == 8 else {2: "paired4", 3: "triplet4"}[m]
                x = mx.random.normal((1, m, 5120)).astype(mx.float16)
                mx.eval(x)
                actual = candidate.qmv(x, *packed, bits=bits, group_size=group,
                                       variant=variant, output_splits=widths)
                self.assertIsInstance(actual, tuple)
                self.assertEqual(len(actual), len(parts))
                for index, (output, part) in enumerate(zip(actual, parts)):
                    with self.subTest(bits=bits, widths=widths, m=m, part=index):
                        self.assert_exact(output, mx.quantized_matmul(
                            x, *part, transpose=True, bits=bits, group_size=group))

    def test_partitioned_strided_compiled_replay(self):
        widths = (8, 16, 48)
        for bits, group in ((4, 32), (8, 64)):
            arrays = tuple(a[::2] for a in self.arrays(bits, group, n=2 * sum(widths)))
            for m in (2, 3):
                variants = ["shared"] + ([{2: "paired4", 3: "triplet4"}[m]] if bits == 4 else [])
                for variant in variants:
                    def fn(x):
                        return candidate.qmv(x, *arrays, bits=bits, group_size=group,
                                             variant=variant, output_splits=widths)

                    compiled = mx.compile(fn)
                    for _ in range(2):
                        x = mx.random.normal((1, 2 * m, 1024)).astype(mx.float16)[:, ::2, ::2]
                        mx.eval(x, *arrays)
                        expected = mx.split(mx.quantized_matmul(
                            x, *arrays, transpose=True, bits=bits, group_size=group), (8, 24), axis=-1)
                        for actual in (fn(x), compiled(x)):
                            for output, reference in zip(actual, expected):
                                with self.subTest(bits=bits, m=m, variant=variant):
                                    self.assert_exact(output, reference)

    def test_partitioned_rejects_invalid_widths_and_modes(self):
        arrays = self.arrays(4, 32, n=24)
        x = mx.zeros((2, 512), dtype=mx.float16)
        for widths in ((), [8, 16], (8,), (8, 0, 16), (-8, 32), (4, 20), (8, 16.0), (True, 23)):
            with self.subTest(widths=widths), self.assertRaisesRegex(ValueError, "output_splits"):
                candidate.qmv(x, *arrays, bits=4, group_size=32,
                              variant="paired4", output_splits=widths)
        for variant, m in (("flat", 2), ("stream4", 2), ("quartet4", 2), ("shared", 1), ("shared", 4)):
            with self.subTest(variant=variant, m=m), self.assertRaisesRegex(ValueError, "output_splits"):
                candidate.qmv(mx.zeros((m, 512), mx.float16), *arrays, bits=4,
                              group_size=32, variant=variant, output_splits=(8, 16))


if __name__ == "__main__":
    unittest.main()

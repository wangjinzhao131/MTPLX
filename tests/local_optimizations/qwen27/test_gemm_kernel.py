"""Exact GPU checks for the experimental native-arithmetic QMM tile sweep."""

import unittest

import mlx.core as mx

from mtplx.local_optimizations.qwen27 import gemm_kernel as candidate


def quantized(n, k, bits, group_size):
    weight = (mx.random.normal((n, k)) * 0.125).astype(mx.float16)
    packed = mx.quantize(weight, group_size=group_size, bits=bits)
    mx.eval(packed)
    return packed


class GemmKernelTests(unittest.TestCase):
    def setUp(self):
        mx.random.seed(443)

    def assert_exact(self, actual, expected):
        mx.eval(actual, expected)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(mx.array_equal(actual, expected).item(), {
            "max_abs_error": float(mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))).item()),
            "unequal_elements": int(mx.sum(actual != expected).item()),
        })

    def test_all_tiles_native_and_compiled_without_split_k(self):
        for bits, group in ((4, 32), (8, 64)):
            packed = quantized(1024, 256, bits, group)
            tiles = candidate.TILES + candidate.BM128_TILES + (candidate.Q8_EXTRA_TILES if bits == 8 else ())
            # ceil(M/32)*ceil(N/32) >= 512 excludes native split-K.
            for length in (512, 513):
                x = mx.random.normal((length, 256)).astype(mx.float16)
                mx.eval(x)
                expected = mx.quantized_matmul(x, *packed, transpose=True, group_size=group, bits=bits)
                for tile in tiles:
                    with self.subTest(bits=bits, length=length, tile=tile):
                        def run(value):
                            return candidate.qmm(value, *packed, group_size=group, bits=bits, tile=tile)
                        self.assert_exact(run(x), expected)
                        self.assert_exact(mx.compile(run)(x), expected)

    def test_m_tails_leading_dimensions_and_strides(self):
        for bits, group in ((4, 32), (8, 64)):
            packed = quantized(256, group, bits, group)
            # One quantization group along K also excludes native split-K.
            for shape in ((33, group), (65, group), (127, group),
                          (128, group), (129, group), (1, 3, 33, group)):
                x = mx.random.normal((*shape[:-1], group * 2)).astype(mx.float16)[..., ::2]
                strided = tuple(mx.stack((a, a), axis=-1)[..., 0] for a in packed)
                mx.eval(x, strided)
                expected = mx.quantized_matmul(x, *strided, transpose=True, group_size=group, bits=bits)
                for tile in (candidate.TILES[1], candidate.TILES[-1], *candidate.BM128_TILES):
                    with self.subTest(bits=bits, shape=shape, tile=tile):
                        self.assert_exact(candidate.qmm(x, *strided, group_size=group, bits=bits, tile=tile), expected)

    def test_zero_and_large_input_scale(self):
        packed = quantized(1024, 128, 4, 32)
        x = mx.random.normal((512, 128)).astype(mx.float16)
        mx.eval(x)
        for scale in (0.0, 16.0):
            value = x * scale
            expected = mx.quantized_matmul(value, *packed, transpose=True, group_size=32, bits=4)
            for tile in (candidate.TILES[1], *candidate.BM128_TILES):
                with self.subTest(scale=scale, tile=tile):
                    self.assert_exact(candidate.qmm(value, *packed, group_size=32, bits=4, tile=tile), expected)

    def test_validation(self):
        packed = quantized(128, 64, 4, 32)
        x = mx.zeros((64, 64), dtype=mx.float16)
        cases = [({"tile": (16, 32, 32, 2, 2)}, "unsupported"),
                 ({"tile": candidate.Q8_EXTRA_TILES[0]}, "BK"),
                 ({"bits": 6}, "expected affine")]
        for overrides, error in cases:
            kwargs = {"bits": 4, "group_size": 32, **overrides}
            with self.assertRaisesRegex(ValueError, error):
                candidate.qmm(x, *packed, **kwargs)
        for value in (x[:1], x.astype(mx.float32), x[:, :32]):
            with self.assertRaises(ValueError):
                candidate.qmm(value, *packed, bits=4, group_size=32)
        with self.assertRaisesRegex(ValueError, "N divisible"):
            candidate.qmm(x, *(a[:96] for a in packed), bits=4, group_size=32, tile=candidate.TILES[2])


if __name__ == "__main__":
    unittest.main()

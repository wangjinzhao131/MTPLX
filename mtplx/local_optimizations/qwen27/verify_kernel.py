"""Experimental small-batch affine GEMV; no automatic model installation.

The flat control preserves native qmv_fast arithmetic. The shared candidate
loads each packed weight once for all input vectors in a threadgroup, while
retaining each vector's load_vector/qdot/simd_sum arithmetic. GPU exactness and
whole-model speed must be measured before adopting either implementation.

Derived from MLX v0.32.2 mlx/backend/metal/kernels/quantized.h:
https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/quantized.h
Commit: 1f8e74e3f12f31365464a6867c6579f0e9b29d85
SPDX-License-Identifier: MIT

MIT License

Copyright © 2023-2024 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import mlx.core as mx

SOURCE_COMMIT = "1f8e74e3f12f31365464a6867c6579f0e9b29d85"
SOURCE_SHA256 = "2a007016da606afe569adb9adcc05e00f14558ad2e094bcb4f8974beb53c316f"
VARIANTS = ("flat", "shared", "stream2", "stream4", "paired4", "paired8", "quartet4", "triplet4")

# Only the target's q4 and q8 branches of MLX load_vector/qdot are retained.
_HEADER = r"""
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector(const device T* x, thread U* x_thread) {
  U sum = 0;
  if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  } else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }
  return sum;
}
"""

_QDOT = r"""
template <typename U, int values_per_thread, int bits>
inline U qdot_ADDRESS(
    const ADDRESS uint8_t* w,
    const thread U* x_thread,
    U scale, U bias, U sum) {
  U accum = 0;
  if (bits == 4) {
    const ADDRESS uint16_t* ws = (const ADDRESS uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  } else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }
  return scale * accum + sum * bias;
}
"""
_HEADER += _QDOT.replace("ADDRESS", "device") + _QDOT.replace("ADDRESS", "thread")
_HEADER += r"""
template <typename U, int values_per_thread>
inline U qdot_packed4(
    uint32_t low, uint32_t high, const thread U* x_thread,
    U scale, U bias, U sum) {
  U accum = 0;
  for (int i = 0; i < (values_per_thread / 4); i++) {
    // Explicit halves avoid aliasing the private uint32 buffer as uint16.
    uint16_t w = static_cast<uint16_t>((i < 2 ? low : high) >> (16 * (i % 2)));
    accum +=
        (x_thread[4 * i] * (w & 0x000f) +
         x_thread[4 * i + 1] * (w & 0x00f0) +
         x_thread[4 * i + 2] * (w & 0x0f00) +
         x_thread[4 * i + 3] * (w & 0xf000));
  }
  return scale * accum + sum * bias;
}
"""

_FLAT = r"""
constexpr int V = 64 / BITS;
constexpr int BLOCK = V * 32;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
// Native already dispatches (M, N/8): this only flattens its group ordering.
const int vec = threadgroup_position_in_grid.x % M;
const int row0 = threadgroup_position_in_grid.x / M * 8 + sg * 4;
const int bytes_per_row = K * BITS / 8;
const int groups_per_row = K / GROUP;
const device uint8_t* ws = (const device uint8_t*)weight
    + row0 * bytes_per_row + lane * 8;
const device T* sl = scales + row0 * groups_per_row + lane / (GROUP / V);
const device T* bl = biases + row0 * groups_per_row + lane / (GROUP / V);
const device T* xv = x + vec * K + lane * V;
float x_thread[V];
float result[4] = {0};
for (int k = 0; k < K; k += BLOCK) {
  float sum = load_vector<T, float, V, BITS>(xv, x_thread);
  for (int r = 0; r < 4; r++) {
    float s = sl[r * groups_per_row];
    float b = bl[r * groups_per_row];
    result[r] += qdot_device<float, V, BITS>(
        ws + r * bytes_per_row, x_thread, s, b, sum);
  }
  ws += BLOCK * BITS / 8;
  sl += BLOCK / GROUP;
  bl += BLOCK / GROUP;
  xv += BLOCK;
}
for (int r = 0; r < 4; r++) {
  result[r] = simd_sum(result[r]);
  if (lane == 0) y[vec * N + row0 + r] = static_cast<T>(result[r]);
}
"""

_SHARED = r"""
constexpr int V = 64 / BITS;
constexpr int BLOCK = V * 32;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int row0 = threadgroup_position_in_grid.x * 8 + sg * 4;
const int words_per_row = K * BITS / 32;
const int groups_per_row = K / GROUP;
const device uint32_t* ws = weight + row0 * words_per_row + lane * 2;
const device T* sl = scales + row0 * groups_per_row + lane / (GROUP / V);
const device T* bl = biases + row0 * groups_per_row + lane / (GROUP / V);
float x_thread[M][V];
float sums[M];
float result[M][4] = {{0}};
for (int k = 0; k < K; k += BLOCK) {
  for (int v = 0; v < M; v++) {
    sums[v] = load_vector<T, float, V, BITS>(
        x + v * K + k + lane * V, x_thread[v]);
  }
  for (int r = 0; r < 4; r++) {
    uint32_t packed[2] = {ws[r * words_per_row], ws[r * words_per_row + 1]};
    float s = sl[r * groups_per_row];
    float b = bl[r * groups_per_row];
    for (int v = 0; v < M; v++) {
      if constexpr (BITS == 4) {
        result[v][r] += qdot_packed4<float, V>(
            packed[0], packed[1], x_thread[v], s, b, sums[v]);
      } else {
        result[v][r] += qdot_thread<float, V, BITS>(
            (const thread uint8_t*)packed, x_thread[v], s, b, sums[v]);
      }
    }
  }
  ws += BLOCK * BITS / 32;
  sl += BLOCK / GROUP;
  bl += BLOCK / GROUP;
}
for (int v = 0; v < M; v++) {
  for (int r = 0; r < 4; r++) {
    result[v][r] = simd_sum(result[v][r]);
    if (lane == 0) y[v * N + row0 + r] = static_cast<T>(result[v][r]);
  }
}
"""

_STREAM = r"""
constexpr int V = 64 / BITS;
constexpr int BLOCK = V * 32;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int row0 = threadgroup_position_in_grid.x * (2 * ROWS) + sg * ROWS;
const int words_per_row = K * BITS / 32;
const int groups_per_row = K / GROUP;
const device uint32_t* ws = weight + row0 * words_per_row + lane * 2;
const device T* sl = scales + row0 * groups_per_row + lane / (GROUP / V);
const device T* bl = biases + row0 * groups_per_row + lane / (GROUP / V);
float result[M][ROWS] = {{0}};
for (int k = 0; k < K; k += BLOCK) {
  // Stream x vectors so only V activation temporaries remain live, not M*V.
  uint32_t packed[ROWS][2];
  float scale[ROWS], bias[ROWS];
  _Pragma("clang loop unroll(full)")
  for (int r = 0; r < ROWS; r++) {
    packed[r][0] = ws[r * words_per_row];
    packed[r][1] = ws[r * words_per_row + 1];
    scale[r] = sl[r * groups_per_row];
    bias[r] = bl[r * groups_per_row];
  }
  _Pragma("clang loop unroll(full)")
  for (int v = 0; v < M; v++) {
    float x_thread[V];
    float sum = load_vector<T, float, V, BITS>(
        x + v * K + k + lane * V, x_thread);
    _Pragma("clang loop unroll(full)")
    for (int r = 0; r < ROWS; r++) {
      if constexpr (BITS == 4) {
        result[v][r] += qdot_packed4<float, V>(
            packed[r][0], packed[r][1], x_thread, scale[r], bias[r], sum);
      } else {
        result[v][r] += qdot_thread<float, V, BITS>(
            (const thread uint8_t*)packed[r], x_thread, scale[r], bias[r], sum);
      }
    }
  }
  ws += BLOCK * BITS / 32;
  sl += BLOCK / GROUP;
  bl += BLOCK / GROUP;
}
_Pragma("clang loop unroll(full)")
for (int v = 0; v < M; v++) {
  _Pragma("clang loop unroll(full)")
  for (int r = 0; r < ROWS; r++) {
    result[v][r] = simd_sum(result[v][r]);
    if (lane == 0) y[v * N + row0 + r] = static_cast<T>(result[v][r]);
  }
}
"""

_PAIRED_HEADER = r"""
inline float2 qdot_pair4(
    uint32_t low, uint32_t high,
    const thread float* x0, const thread float* x1,
    float scale, float bias, float sum0, float sum1) {
  float accum0 = 0, accum1 = 0;
  for (int i = 0; i < 4; i++) {
    uint16_t w = static_cast<uint16_t>((i < 2 ? low : high) >> (16 * (i % 2)));
    // Share each converted nibble without changing either vector's dot order.
    float w0 = static_cast<float>(w & 0x000f);
    float w1 = static_cast<float>(w & 0x00f0);
    float w2 = static_cast<float>(w & 0x0f00);
    float w3 = static_cast<float>(w & 0xf000);
    accum0 += (x0[4 * i] * w0 + x0[4 * i + 1] * w1 +
               x0[4 * i + 2] * w2 + x0[4 * i + 3] * w3);
    accum1 += (x1[4 * i] * w0 + x1[4 * i + 1] * w1 +
               x1[4 * i + 2] * w2 + x1[4 * i + 3] * w3);
  }
  return float2(scale * accum0 + sum0 * bias,
                scale * accum1 + sum1 * bias);
}
"""

_PAIRED = r"""
constexpr int V = 16;
constexpr int BLOCK = V * 32;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int row0 = threadgroup_position_in_grid.x * 8 + sg * 4;
const int words_per_row = K / 8;
const int groups_per_row = K / GROUP;
const device uint32_t* ws = weight + row0 * words_per_row + lane * 2;
const device T* sl = scales + row0 * groups_per_row + lane / (GROUP / V);
const device T* bl = biases + row0 * groups_per_row + lane / (GROUP / V);
float result0[4] = {0}, result1[4] = {0};
for (int k = 0; k < K; k += BLOCK) {
  float x0[V], x1[V];
  float sum0 = load_vector<T, float, V, 4>(x + k + lane * V, x0);
  float sum1 = load_vector<T, float, V, 4>(x + K + k + lane * V, x1);
  _Pragma("clang loop unroll(full)")
  for (int r = 0; r < 4; r++) {
    float2 dot = qdot_pair4(ws[r * words_per_row], ws[r * words_per_row + 1],
                          x0, x1, sl[r * groups_per_row], bl[r * groups_per_row],
                          sum0, sum1);
    result0[r] += dot.x;
    result1[r] += dot.y;
  }
  ws += BLOCK / 8;
  sl += BLOCK / GROUP;
  bl += BLOCK / GROUP;
}
_Pragma("clang loop unroll(full)")
for (int r = 0; r < 4; r++) {
  result0[r] = simd_sum(result0[r]);
  result1[r] = simd_sum(result1[r]);
  if (lane == 0) {
    y[row0 + r] = static_cast<T>(result0[r]);
    y[N + row0 + r] = static_cast<T>(result1[r]);
  }
}
"""

_TRIPLET4_HEADER = r"""
inline float3 qdot_triplet4(
    uint32_t low, uint32_t high,
    const thread float* x0, const thread float* x1, const thread float* x2,
    float scale, float bias, float sum0, float sum1, float sum2) {
  float accum0 = 0, accum1 = 0, accum2 = 0;
  for (int i = 0; i < 4; i++) {
    uint16_t w = static_cast<uint16_t>((i < 2 ? low : high) >> (16 * (i % 2)));
    // Share conversion, retaining each vector's native four-product grouping.
    float w0 = static_cast<float>(w & 0x000f);
    float w1 = static_cast<float>(w & 0x00f0);
    float w2 = static_cast<float>(w & 0x0f00);
    float w3 = static_cast<float>(w & 0xf000);
    accum0 += (x0[4 * i] * w0 + x0[4 * i + 1] * w1 +
               x0[4 * i + 2] * w2 + x0[4 * i + 3] * w3);
    accum1 += (x1[4 * i] * w0 + x1[4 * i + 1] * w1 +
               x1[4 * i + 2] * w2 + x1[4 * i + 3] * w3);
    accum2 += (x2[4 * i] * w0 + x2[4 * i + 1] * w1 +
               x2[4 * i + 2] * w2 + x2[4 * i + 3] * w3);
  }
  return float3(scale * accum0 + sum0 * bias,
                scale * accum1 + sum1 * bias,
                scale * accum2 + sum2 * bias);
}
"""

_TRIPLET4 = r"""
constexpr int V = 16;
constexpr int BLOCK = V * 32;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int row0 = threadgroup_position_in_grid.x * 8 + sg * 4;
const int words_per_row = K / 8;
const int groups_per_row = K / GROUP;
const device uint32_t* ws = weight + row0 * words_per_row + lane * 2;
const device T* sl = scales + row0 * groups_per_row + lane / (GROUP / V);
const device T* bl = biases + row0 * groups_per_row + lane / (GROUP / V);
float result0[4] = {0}, result1[4] = {0}, result2[4] = {0};
for (int k = 0; k < K; k += BLOCK) {
  float x0[V], x1[V], x2[V];
  float sum0 = load_vector<T, float, V, 4>(x + k + lane * V, x0);
  float sum1 = load_vector<T, float, V, 4>(x + K + k + lane * V, x1);
  float sum2 = load_vector<T, float, V, 4>(x + 2 * K + k + lane * V, x2);
  _Pragma("clang loop unroll(full)")
  for (int r = 0; r < 4; r++) {
    float3 dot = qdot_triplet4(ws[r * words_per_row], ws[r * words_per_row + 1],
                             x0, x1, x2, sl[r * groups_per_row], bl[r * groups_per_row],
                             sum0, sum1, sum2);
    result0[r] += dot.x;
    result1[r] += dot.y;
    result2[r] += dot.z;
  }
  ws += BLOCK / 8;
  sl += BLOCK / GROUP;
  bl += BLOCK / GROUP;
}
_Pragma("clang loop unroll(full)")
for (int r = 0; r < 4; r++) {
  result0[r] = simd_sum(result0[r]);
  result1[r] = simd_sum(result1[r]);
  result2[r] = simd_sum(result2[r]);
  if (lane == 0) {
    y[row0 + r] = static_cast<T>(result0[r]);
    y[N + row0 + r] = static_cast<T>(result1[r]);
    y[2 * N + row0 + r] = static_cast<T>(result2[r]);
  }
}
"""

_PAIRED8_HEADER = r"""
inline float2 qdot_pair8(
    uint32_t low, uint32_t high,
    const thread float* x0, const thread float* x1,
    float scale, float bias, float sum0, float sum1) {
  float accum0 = 0, accum1 = 0;
  for (int i = 0; i < 8; i++) {
    uint32_t packed = i < 4 ? low : high;
    float w = static_cast<float>((packed >> (8 * (i % 4))) & 0xff);
    // Preserve each native q8 dot's sequential accumulation; share conversion.
    accum0 += x0[i] * w;
    accum1 += x1[i] * w;
  }
  return float2(scale * accum0 + sum0 * bias,
                scale * accum1 + sum1 * bias);
}
"""

# Keep the validated paired4 source unchanged. Q8 uses eight values/lane and
# 256-value K blocks, with the same two packed words and per-row FP32 results.
_PAIRED8 = (_PAIRED
            .replace("constexpr int V = 16;", "constexpr int V = 8;")
            .replace("const int words_per_row = K / 8;", "const int words_per_row = K / 4;")
            .replace("ws += BLOCK / 8;", "ws += BLOCK / 4;")
            .replace("V, 4>", "V, 8>")
            .replace("qdot_pair4(", "qdot_pair8("))

_QUARTET4 = r"""
constexpr int V = 16;
constexpr int BLOCK = V * 32;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int row0 = threadgroup_position_in_grid.x * 8 + sg * 4;
const int words_per_row = K / 8;
const int groups_per_row = K / GROUP;
const device uint32_t* ws = weight + row0 * words_per_row + lane * 2;
const device T* sl = scales + row0 * groups_per_row + lane / (GROUP / V);
const device T* bl = biases + row0 * groups_per_row + lane / (GROUP / V);
float result0[4] = {0}, result1[4] = {0};
for (int k = 0; k < K; k += BLOCK) {
  float dot0[4] = {0}, dot1[4] = {0};
  float sum0 = 0, sum1 = 0;
  _Pragma("clang loop unroll(full)")
  for (int word = 0; word < 2; word++) {
    uint32_t packed[4];
    _Pragma("clang loop unroll(full)")
    for (int r = 0; r < 4; r++) packed[r] = ws[r * words_per_row + word];
    _Pragma("clang loop unroll(full)")
    for (int part = 0; part < 2; part++) {
      const int offset = k + lane * V + word * 8 + part * 4;
      const device T* p0 = x + offset;
      const device T* p1 = x + K + offset;
      // Copy the native T-operand addition before conversion to float. Summing
      // the normalized float4 below would change FP16 bias-sum arithmetic.
      sum0 += p0[0] + p0[1] + p0[2] + p0[3];
      sum1 += p1[0] + p1[1] + p1[2] + p1[3];
      float4 x0 = float4(p0[0], p0[1] / 16.0f,
                         p0[2] / 256.0f, p0[3] / 4096.0f);
      float4 x1 = float4(p1[0], p1[1] / 16.0f,
                         p1[2] / 256.0f, p1[3] / 4096.0f);
      _Pragma("clang loop unroll(full)")
      for (int r = 0; r < 4; r++) {
        uint16_t w = static_cast<uint16_t>(packed[r] >> (16 * part));
        float w0 = static_cast<float>(w & 0x000f);
        float w1 = static_cast<float>(w & 0x00f0);
        float w2 = static_cast<float>(w & 0x0f00);
        float w3 = static_cast<float>(w & 0xf000);
        dot0[r] += (x0.x * w0 + x0.y * w1 + x0.z * w2 + x0.w * w3);
        dot1[r] += (x1.x * w0 + x1.y * w1 + x1.z * w2 + x1.w * w3);
      }
    }
  }
  _Pragma("clang loop unroll(full)")
  for (int r = 0; r < 4; r++) {
    float scale = sl[r * groups_per_row], bias = bl[r * groups_per_row];
    result0[r] += scale * dot0[r] + sum0 * bias;
    result1[r] += scale * dot1[r] + sum1 * bias;
  }
  ws += BLOCK / 8;
  sl += BLOCK / GROUP;
  bl += BLOCK / GROUP;
}
_Pragma("clang loop unroll(full)")
for (int r = 0; r < 4; r++) {
  result0[r] = simd_sum(result0[r]);
  result1[r] = simd_sum(result1[r]);
  if (lane == 0) {
    y[row0 + r] = static_cast<T>(result0[r]);
    y[N + row0 + r] = static_cast<T>(result1[r]);
  }
}
"""

_KERNELS = {}


def _partitioned_source(source, widths):
    header = "\ninline uint partitioned_index(uint row, uint vector, uint vectors) {\n"
    start = 0
    for index, width in enumerate(widths):
        condition = f"if (row < {start + width}) " if index < len(widths) - 1 else ""
        header += f"  {condition}return vectors * {start} + vector * {width} + row - {start};\n"
        start += width
    header += "}\n"
    for address, vector in (("v * N + row0 + r", "v"), ("row0 + r", "0"),
                            ("N + row0 + r", "1"), ("2 * N + row0 + r", "2")):
        source = source.replace(f"y[{address}]", f"y[partitioned_index(row0 + r, {vector}, M)]")
    return header, source


def _kernel(variant, output_splits=None):
    key = variant if output_splits is None else (variant, output_splits)
    if key not in _KERNELS:
        header = _HEADER + {"paired4": _PAIRED_HEADER, "paired8": _PAIRED8_HEADER,
                            "triplet4": _TRIPLET4_HEADER}.get(variant, "")
        source = {"flat": _FLAT, "shared": _SHARED,
                  "stream2": _STREAM, "stream4": _STREAM,
                  "paired4": _PAIRED, "paired8": _PAIRED8,
                  "quartet4": _QUARTET4, "triplet4": _TRIPLET4}[variant]
        suffix = ""
        if output_splits is not None:
            extra_header, source = _partitioned_source(source, output_splits)
            header += extra_header
            suffix = "_parts_" + "_".join(map(str, output_splits))
        _KERNELS[key] = mx.fast.metal_kernel(
            name="qwen38_verify_" + variant + suffix,
            input_names=["x", "weight", "scales", "biases"],
            output_names=["y"],
            header=header, source=source,
            ensure_row_contiguous=True)
    return _KERNELS[key]


def qmv(x, weight, scales, biases, *, group_size, bits, variant="shared", output_splits=None):
    """FP16 affine x @ W.T for 1–4 vectors and native-fast aligned K/N.

    All leading x dimensions flatten into M. Weights and quantization parameters
    are unchanged; noncontiguous inputs are copied by the custom-kernel API.
    The paired4/quartet4 experiments require M=2 q4/group32; paired8 needs q8/group64.
    The triplet4 experiment requires M=3 q4/group32.
    Optional output_splits is a tuple of projection widths for M2/3 paired4,
    triplet4 or shared. It returns contiguous part views of one flat buffer.
    """
    if variant not in VARIANTS:
        raise ValueError("unsupported verify GEMV variant")
    if (bits, group_size) not in ((4, 32), (8, 64)):
        raise ValueError("expected affine q4/group32 or q8/group64")
    if x.ndim < 2 or weight.ndim != 2 or scales.ndim != 2 or biases.ndim != 2:
        raise ValueError("expected x rank >= 2 and 2D quantized tensors")
    if x.dtype != mx.float16 or weight.dtype != mx.uint32 or any(
            a.dtype != mx.float16 for a in (scales, biases)):
        raise ValueError("expected FP16 x/scales/biases and uint32 packed weights")
    k, n = x.shape[-1], weight.shape[0]
    m = x.size // k if k else 0
    if variant == "paired4" and (m, bits, group_size) != (2, 4, 32):
        raise ValueError("paired4 requires M=2 and affine q4/group32")
    if variant == "paired8" and (m, bits, group_size) != (2, 8, 64):
        raise ValueError("paired8 requires M=2 and affine q8/group64")
    if variant == "quartet4" and (m, bits, group_size) != (2, 4, 32):
        raise ValueError("quartet4 requires M=2 and affine q4/group32")
    if variant == "triplet4" and (m, bits, group_size) != (3, 4, 32):
        raise ValueError("triplet4 requires M=3 and affine q4/group32")
    if not 1 <= m <= 4 or n < 8 or n % 8 or k < 2048 // bits or k % (2048 // bits):
        raise ValueError("expected M in 1..4, N divisible by 8, and native-fast aligned K")
    if weight.shape[1] != k * bits // 32 or scales.shape != (n, k // group_size) or biases.shape != scales.shape:
        raise ValueError("quantized tensor shapes do not match x")
    if output_splits is not None:
        if (variant not in ("paired4", "triplet4", "shared") or m not in (2, 3)
                or not isinstance(output_splits, tuple) or not output_splits
                or any(type(width) is not int or width <= 0 or width % 8 for width in output_splits)
                or sum(output_splits) != n):
            raise ValueError("output_splits requires M2/3 paired4/triplet4/shared and positive widths divisible by 8 summing to N")
    rows = 2 if variant == "stream2" else 4
    template = [("T", mx.float16), ("GROUP", group_size), ("BITS", bits),
                ("M", m), ("N", n), ("K", k)]
    if variant in ("stream2", "stream4"):
        template.append(("ROWS", rows))
    output = _kernel(variant, output_splits)(
        inputs=[x, weight, scales, biases],
        template=template,
        grid=(n // (2 * rows) * (m if variant == "flat" else 1) * 64, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(*x.shape[:-1], n) if output_splits is None else (m * n,)],
        output_dtypes=[mx.float16])[0]
    if output_splits is None:
        return output
    parts, start = [], 0
    for width in output_splits:
        parts.append(output[start:start + m * width].reshape(*x.shape[:-1], width))
        start += m * width
    return tuple(parts)

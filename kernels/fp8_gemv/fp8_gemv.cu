// SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
// SPDX-License-Identifier: Apache-2.0
//
// fp8_gemv.cu — matrix-vector product of FP8 (E4M3) weights × BF16 activations (for batch-1 decoding)
//
//   y[m, n] = act( scale[n] * sum_k fp8(W[n, k]) * x[m, k] )      (accumulation in FP32, output in BF16)
//
// - Weight W is row-major [N, K], same as nn.Linear. 1 byte per element (E4M3). Scale is FP32 per output channel.
// - One warp handles ROWS rows. Each lane reads 16 elements at a time (16 bytes of weights = one uint4),
//   and reuses x at the same position (16 BF16 = two uint4) across the ROWS rows. Finally sums within the warp.
// - Weights are read only once per token, so they are streamed without caching in L1 (ld.global.cs).
// - To fit in a CUDA graph, no host synchronization or memory allocation (output is allocated with the PyTorch allocator).
// - act: 0 = none, 1 = relu² (after fc1 in the Cosmos3-Edge MLP).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace {

constexpr int kWarps = 4;  // 128 threads per block

// Convert 32 bits packing 4 E4M3 values into 4 FP32 values (hardware cvt.f16x2.e4m3x2 on sm_89 and later)
__device__ __forceinline__ void fp8x4_to_f32(uint32_t v, float* f) {
  __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(v & 0xFFFFu), __NV_E4M3);
  __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(v >> 16), __NV_E4M3);
  float2 a = __half22float2(*reinterpret_cast<__half2*>(&lo));
  float2 b = __half22float2(*reinterpret_cast<__half2*>(&hi));
  f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y;
}

// Convert 32 bits packing 2 BF16 values into 2 FP32 values (BF16 is the upper 16 bits of FP32)
__device__ __forceinline__ void bf16x2_to_f32(uint32_t v, float* f) {
  f[0] = __uint_as_float(v << 16);
  f[1] = __uint_as_float(v & 0xFFFF0000u);
}

template <int ROWS, int ACT>
__global__ void __launch_bounds__(kWarps * 32)
fp8_gemv_kernel(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ w,
                const float* __restrict__ scale, __nv_bfloat16* __restrict__ y, int N, int K) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * kWarps + warp) * ROWS;
  if (row0 >= N) return;
  const int nchunk = K >> 4;  // number of 16-element chunks
  const uint4* xv = reinterpret_cast<const uint4*>(x);
  const uint4* wr[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    // when N is not a multiple of ROWS, re-read the last row (result is not written)
    const int row = min(row0 + r, N - 1);
    wr[r] = reinterpret_cast<const uint4*>(w + static_cast<size_t>(row) * K);
  }
  float acc[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;

#pragma unroll 2
  for (int c = lane; c < nchunk; c += 32) {
    // issue all weight reads first, then compute (overlap reads to fill the bandwidth)
    uint4 wv[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) wv[r] = __ldcs(wr[r] + c);
    const uint4 xa = __ldg(xv + 2 * c);
    const uint4 xb = __ldg(xv + 2 * c + 1);
    float xf[16];
    bf16x2_to_f32(xa.x, xf + 0);  bf16x2_to_f32(xa.y, xf + 2);
    bf16x2_to_f32(xa.z, xf + 4);  bf16x2_to_f32(xa.w, xf + 6);
    bf16x2_to_f32(xb.x, xf + 8);  bf16x2_to_f32(xb.y, xf + 10);
    bf16x2_to_f32(xb.z, xf + 12); bf16x2_to_f32(xb.w, xf + 14);
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      float wf[16];
      fp8x4_to_f32(wv[r].x, wf + 0);
      fp8x4_to_f32(wv[r].y, wf + 4);
      fp8x4_to_f32(wv[r].z, wf + 8);
      fp8x4_to_f32(wv[r].w, wf + 12);
      float s = acc[r];
#pragma unroll
      for (int i = 0; i < 16; ++i) s = fmaf(wf[i], xf[i], s);
      acc[r] = s;
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], o);
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    const int row = row0 + r;
    if (lane == r && row < N) {
      float v = acc[r] * __ldg(scale + row);
      if (ACT == 1) { v = fmaxf(v, 0.f); v = v * v; }
      y[row] = __float2bfloat16_rn(v);
    }
  }
}

template <int ROWS>
void launch_rows(const __nv_bfloat16* x, const uint8_t* w, const float* s, __nv_bfloat16* y,
                 int N, int K, int64_t act, cudaStream_t st) {
  const int rows_per_block = kWarps * ROWS;
  const dim3 grid((N + rows_per_block - 1) / rows_per_block);
  const dim3 block(kWarps * 32);
  if (act == 1) fp8_gemv_kernel<ROWS, 1><<<grid, block, 0, st>>>(x, w, s, y, N, K);
  else fp8_gemv_kernel<ROWS, 0><<<grid, block, 0, st>>>(x, w, s, y, N, K);
}

// Bandwidth ceiling reference: a kernel that only reads, the same way (uint4 with ld.global.cs)
__global__ void __launch_bounds__(256) read_only_kernel(const uint4* __restrict__ p, size_t n, uint32_t* out) {
  uint32_t a = 0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const uint4 v = __ldcs(p + i);
    a ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (a == 0x9e3779b9u) out[0] = a;  // nominal write so the reads are not eliminated
}

}  // namespace

// x: [..., K] BF16, w: [N, K] uint8 (E4M3 bits), scale: [N] FP32 → [..., N] BF16
torch::Tensor fp8_gemv(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& scale,
                       int64_t act, int64_t rows) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && scale.is_cuda(), "fp8_gemv: CUDA tensors required");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "fp8_gemv: x must be BF16");
  TORCH_CHECK(w.scalar_type() == at::kByte && w.dim() == 2 && w.is_contiguous(), "fp8_gemv: w must be contiguous [N, K] uint8");
  TORCH_CHECK(scale.scalar_type() == at::kFloat && scale.is_contiguous(), "fp8_gemv: scale must be FP32");
  const int64_t N = w.size(0), K = w.size(1);
  TORCH_CHECK(x.size(-1) == K, "fp8_gemv: K mismatch");
  TORCH_CHECK(K % 16 == 0, "fp8_gemv: K must be a multiple of 16");
  TORCH_CHECK(scale.numel() == N, "fp8_gemv: scale length does not match N");
  const c10::cuda::CUDAGuard guard(x.device());
  auto xc = x.contiguous();
  const int64_t M = xc.numel() / K;
  auto out_shape = x.sizes().vec();
  out_shape.back() = N;
  auto y = torch::empty(out_shape, x.options());
  auto st = at::cuda::getCurrentCUDAStream();
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr<at::BFloat16>());
  auto* yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>());
  const auto* wp = w.data_ptr<uint8_t>();
  const auto* sp = scale.data_ptr<float>();
  // M > 1 (does not occur with the single candidate used here) launches once per row
  for (int64_t m = 0; m < M; ++m) {
    const auto* xm = xp + m * K;
    auto* ym = yp + m * N;
    switch (rows) {
      case 1: launch_rows<1>(xm, wp, sp, ym, N, K, act, st); break;
      case 2: launch_rows<2>(xm, wp, sp, ym, N, K, act, st); break;
      case 4: launch_rows<4>(xm, wp, sp, ym, N, K, act, st); break;
      case 8: launch_rows<8>(xm, wp, sp, ym, N, K, act, st); break;
      default: TORCH_CHECK(false, "fp8_gemv: rows must be 1/2/4/8");
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// For bandwidth measurement (not used by the model)
void read_only(const torch::Tensor& buf, torch::Tensor& out, int64_t blocks) {
  TORCH_CHECK(buf.is_cuda() && buf.is_contiguous() && buf.nbytes() % 16 == 0);
  const c10::cuda::CUDAGuard guard(buf.device());
  read_only_kernel<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(buf.data_ptr()), buf.nbytes() / 16,
      reinterpret_cast<uint32_t*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

TORCH_LIBRARY(a2e, m) {
  m.def("fp8_gemv(Tensor x, Tensor w, Tensor scale, int act, int rows) -> Tensor");
  m.def("read_only(Tensor buf, Tensor(a!) out, int blocks) -> ()");
}

TORCH_LIBRARY_IMPL(a2e, CUDA, m) {
  m.impl("fp8_gemv", &fp8_gemv);
  m.impl("read_only", &read_only);
}

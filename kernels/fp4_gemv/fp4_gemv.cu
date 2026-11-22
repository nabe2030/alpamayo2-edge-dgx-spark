// SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
// SPDX-License-Identifier: Apache-2.0
//
// fp4_gemv.cu — FP4 (E2M1) weight × BF16 activation matrix-vector product, for batch-1 decoding.
//
//   y[n] = act( gs[n] * sum_b bs[n, b] * sum_{k in b} e2m1(W[n, k]) * x[k] )     (FP32 accumulation, BF16 output)
//
// - Weight format close to NVFP4: two E2M1 values per byte (the low 4 bits hold the even-indexed element), one E4M3
//   scale per 16 elements, one FP32 scale per row. Weights are [N, K/2] uint8, block scales [N, K/16] uint8,
//   row scales [N] FP32.
// - One warp per row. Each lane reads 16 bytes of weights (32 elements, two blocks) and 32 values of x
//   (BF16, four uint4) per iteration.
// - FP4 -> half2 uses __nv_cvt_fp4x2_to_halfraw2. It becomes the hardware instruction cvt.rn.f16x2.e2m1x2 only when
//   built for an architecture-specific target (sm_120a / sm_121a, ...). With plain sm_121 the same code compiles to
//   a software conversion and is not faster (build.sh uses sm_121a).
// - As in fp8_gemv.cu, there is no host synchronization and no allocation other than the output, so the op can be
//   captured in a CUDA graph.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdint>

namespace {

constexpr int kWarps = 4;

__device__ __forceinline__ void bf16x2_to_f32(uint32_t v, float* f) {
  f[0] = __uint_as_float(v << 16);
  f[1] = __uint_as_float(v & 0xFFFF0000u);
}

// dot product of 4 bytes (8 FP4 values) with 8 values of x
__device__ __forceinline__ float dot8(uint32_t w, const float* xf) {
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    __half2_raw h = __nv_cvt_fp4x2_to_halfraw2(static_cast<__nv_fp4x2_storage_t>((w >> (8 * i)) & 0xFFu), __NV_E2M1);
    float2 f = __half22float2(*reinterpret_cast<__half2*>(&h));
    s = fmaf(f.x, xf[2 * i], s);
    s = fmaf(f.y, xf[2 * i + 1], s);
  }
  return s;
}

__device__ __forceinline__ float e4m3_to_f32(uint8_t v) {
  __half_raw h = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(v), __NV_E4M3);
  return __half2float(*reinterpret_cast<__half*>(&h));
}

template <int ACT>
__global__ void __launch_bounds__(kWarps * 32)
fp4_gemv_kernel(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ bs,
                const float* __restrict__ gs, __nv_bfloat16* __restrict__ y, int N, int K) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kWarps + warp;
  if (row >= N) return;
  const int nchunk = K >> 5;  // chunks of 32 elements (16 bytes)
  const uint4* wr = reinterpret_cast<const uint4*>(w + static_cast<size_t>(row) * (K >> 1));
  const uint16_t* br = reinterpret_cast<const uint16_t*>(bs + static_cast<size_t>(row) * (K >> 4));
  const uint4* xv = reinterpret_cast<const uint4*>(x);
  float acc = 0.f;
#pragma unroll 2
  for (int c = lane; c < nchunk; c += 32) {
    const uint4 wv = __ldcs(wr + c);
    const uint16_t sc = __ldcs(br + c);
    float xf[32];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint4 xa = __ldg(xv + 4 * c + j);
      bf16x2_to_f32(xa.x, xf + 8 * j + 0);
      bf16x2_to_f32(xa.y, xf + 8 * j + 2);
      bf16x2_to_f32(xa.z, xf + 8 * j + 4);
      bf16x2_to_f32(xa.w, xf + 8 * j + 6);
    }
    const float s0 = dot8(wv.x, xf + 0) + dot8(wv.y, xf + 8);
    const float s1 = dot8(wv.z, xf + 16) + dot8(wv.w, xf + 24);
    acc = fmaf(s0, e4m3_to_f32(sc & 0xFFu), acc);
    acc = fmaf(s1, e4m3_to_f32(sc >> 8), acc);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) {
    float v = acc * __ldg(gs + row);
    if (ACT == 1) { v = fmaxf(v, 0.f); v = v * v; }
    y[row] = __float2bfloat16_rn(v);
  }
}

}  // namespace

// x: [..., K] BF16, w: [N, K/2] uint8, bs: [N, K/16] uint8 (E4M3), gs: [N] FP32 -> [..., N] BF16
torch::Tensor fp4_gemv(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& bs, const torch::Tensor& gs,
                       int64_t act) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && bs.is_cuda() && gs.is_cuda(), "fp4_gemv: CUDA tensors required");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "fp4_gemv: x must be BF16");
  TORCH_CHECK(w.scalar_type() == at::kByte && w.dim() == 2 && w.is_contiguous(), "fp4_gemv: w must be a contiguous [N, K/2] uint8 tensor");
  TORCH_CHECK(bs.scalar_type() == at::kByte && bs.is_contiguous(), "fp4_gemv: bs must be contiguous uint8");
  TORCH_CHECK(gs.scalar_type() == at::kFloat && gs.is_contiguous(), "fp4_gemv: gs must be FP32");
  const int64_t N = w.size(0), K = w.size(1) * 2;
  TORCH_CHECK(x.size(-1) == K, "fp4_gemv: K does not match");
  TORCH_CHECK(K % 32 == 0, "fp4_gemv: K must be a multiple of 32");
  TORCH_CHECK(bs.numel() == N * (K / 16) && gs.numel() == N, "fp4_gemv: scale sizes do not match");
  const c10::cuda::CUDAGuard guard(x.device());
  auto xc = x.contiguous();
  const int64_t M = xc.numel() / K;
  auto out_shape = x.sizes().vec();
  out_shape.back() = N;
  auto y = torch::empty(out_shape, x.options());
  auto st = at::cuda::getCurrentCUDAStream();
  const dim3 grid((N + kWarps - 1) / kWarps), block(kWarps * 32);
  for (int64_t m = 0; m < M; ++m) {
    const auto* xm = reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr<at::BFloat16>()) + m * K;
    auto* ym = reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()) + m * N;
    if (act == 1) fp4_gemv_kernel<1><<<grid, block, 0, st>>>(xm, w.data_ptr<uint8_t>(), bs.data_ptr<uint8_t>(), gs.data_ptr<float>(), ym, N, K);
    else fp4_gemv_kernel<0><<<grid, block, 0, st>>>(xm, w.data_ptr<uint8_t>(), bs.data_ptr<uint8_t>(), gs.data_ptr<float>(), ym, N, K);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

TORCH_LIBRARY(a2e4, m) {
  m.def("fp4_gemv(Tensor x, Tensor w, Tensor bs, Tensor gs, int act) -> Tensor");
}

TORCH_LIBRARY_IMPL(a2e4, CUDA, m) {
  m.impl("fp4_gemv", &fp4_gemv);
}

# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# a2e_fp4.py — read the weights of the 28 body layers in FP4 (close to NVFP4) during CoT decoding, with plain
#   round-to-nearest quantization. The FP4 GEMV op (kernels/fp4_gemv) is loaded here; a2e_fp4q.py builds on this file
#   (GPTQ and other ways of choosing the 4-bit values) and is the one used in the article.
#   Uses DecodeStepQ of a2e_fp8.py as is and replaces only the layer projections (qkv, o, fc1, fc2) with Q4Linear.
#   The output layer is FP8 or BF16.
#   wide_eval.py --variant-module a2e_fp4.py adds the variants fp4body / fp4body_fp8head.
#   Quantization: one E4M3 scale per block of 16 elements and one FP32 scale per row.
#     gs[n] = max|W[n]| / (6 * 448), bs[n,b] = e4m3(max|W[n,b]| / 6 / gs[n]), q = e2m1(W / (bs * gs)) (nearest value)
import os
from pathlib import Path

import torch

import a2e_fp8 as Q

# built by kernels/fp4_gemv/build.sh (env.sh sets A2E_FP4_SO)
SO4 = os.environ.get("A2E_FP4_SO", str(Path(os.environ.get("A2E_ROOT", ".")) / "build/fp4_gemv/v1/fp4_gemv.so"))
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
_loaded = False


def load_ops():
    global _loaded
    if not _loaded:
        torch.ops.load_library(SO4)

        @torch.library.register_fake("a2e4::fp4_gemv")
        def _(x, w, bs, gs, act):
            return x.new_empty((*x.shape[:-1], w.shape[0]))
        _loaded = True


def quantize_nvfp4(weight: torch.Tensor, chunk: int = 4096):
    """[N, K] → (uint8 [N, K/2], uint8 [N, K/16] E4M3, FP32 [N])"""
    w = weight.detach()
    N, K = w.shape
    assert K % 32 == 0
    qw = torch.empty(N, K // 2, dtype=torch.uint8, device=w.device)
    bs = torch.empty(N, K // 16, dtype=torch.uint8, device=w.device)
    gs = torch.empty(N, dtype=torch.float32, device=w.device)
    mid = _MID.to(w.device)
    for i in range(0, N, chunk):
        x = w[i : i + chunk].float()
        g = (x.abs().amax(dim=1) / (6 * 448)).clamp(min=1e-12)
        xb = x.view(x.shape[0], K // 16, 16)
        b = (xb.abs().amax(dim=2) / 6 / g[:, None]).clamp(min=2.0**-9, max=448).to(torch.float8_e4m3fn)
        scale = b.float() * g[:, None]
        y = xb / scale[:, :, None]
        code = torch.bucketize(y.abs().clamp(max=6.0), mid).to(torch.uint8) | ((y < 0).to(torch.uint8) << 3)
        code = code.view(x.shape[0], K)
        qw[i : i + chunk] = code[:, 0::2] | (code[:, 1::2] << 4)
        bs[i : i + chunk] = b.view(torch.uint8)
        gs[i : i + chunk] = g
    return qw, bs, gs


def dequantize_nvfp4(qw, bs, gs):
    N = qw.shape[0]
    lo, hi = qw & 0xF, qw >> 4
    code = torch.stack([lo, hi], dim=-1).view(N, -1).long()
    val = _E2M1.to(qw.device)[code & 7] * torch.where(code >= 8, -1.0, 1.0)
    K = val.shape[1]
    sc = bs.view(torch.float8_e4m3fn).float() * gs[:, None]
    return (val.view(N, K // 16, 16) * sc[:, :, None]).view(N, K)


class Q4Linear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, act: int = 0) -> None:
        super().__init__()
        qw, bs, gs = quantize_nvfp4(weight)
        self.register_buffer("w", qw, persistent=False)
        self.register_buffer("bs", bs, persistent=False)
        self.register_buffer("gs", gs, persistent=False)
        self.act = int(act)

    def forward(self, x):
        return torch.ops.a2e4.fp4_gemv(x.to(torch.bfloat16), self.w, self.bs, self.gs, self.act)

    def nbytes(self):
        return self.w.numel() + self.bs.numel() + self.gs.numel() * 4


class Q4Layer(torch.nn.Module):
    """Same layout as Q.QLayer (qkv, o, fc1, fc2) in FP4. Usable as is from decoder_block_q."""

    def __init__(self, layer) -> None:
        super().__init__()
        attn, mlp = layer.self_attn, layer.mlp
        qkv = torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0)
        self.nq, self.nkv = attn.q_proj.out_features, attn.k_proj.out_features
        self.qkv = Q4Linear(qkv)
        self.o = Q4Linear(attn.o_proj.weight)
        self.fc1 = Q4Linear(mlp.fc1.weight, act=1)
        self.fc2 = Q4Linear(mlp.fc2.weight)


def build_step4(vlm, head: str = "fp8"):
    load_ops()
    Q.load_ops()
    with torch.no_grad():
        layers = torch.nn.ModuleList(Q4Layer(l) for l in vlm.model.language_model.layers)
        qhead = Q.QLinear(vlm.lm_head.weight, 1) if head == "fp8" else None
    return Q.DecodeStepQ(vlm, layers, qhead)


def _v(head):
    def make(vlm, handle):
        return Q.compiled(build_step4(vlm, head), handle)
    return make


VARIANTS = {"fp4body": _v("bf16"), "fp4body_fp8head": _v("fp8")}

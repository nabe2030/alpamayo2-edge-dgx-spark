#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# bench_fp8_gemv.py <fp8_gemv.so> [--iters 200] [--out result.json]
# Standalone benchmark that uses neither the dataset nor the model (synthetic data).
#   In the matrix-vector product shapes of one decode token of the Alpamayo 2 Edge language model (text side of Cosmos3-Edge: hidden 2048, MLP 9216, attention 16+8+8 heads × 128,
#   vocabulary 135168), measures
#   1. correctness: compared with values computed in PyTorch from the same FP8 weights converted back to FP32 (relative error)
#   2. speed: the kernel / PyTorch BF16 F.linear / a kernel that only reads the same amount (bandwidth reference). Measured with CUDA events
#   Weights are normal random numbers (seed 0). Per-row scale is max|w| / 448 (the E4M3 maximum).
#   In real decoding the weights of all layers (about 1.5 GB) are read once per token, so weights do not stay in cache.
#   Reading the same weights repeatedly would put them in L2 and look faster, so weight copies are made until the total exceeds 512 MB and read in rotation.
import argparse
import json

import torch
import torch.nn.functional as F

SHAPES = {  # name: (N, K, act)  act 0 = none, 1 = relu²
    "qkv": (2048 + 1024 + 1024, 2048, 0),
    "o": (2048, 2048, 0),
    "fc1": (9216, 2048, 1),
    "fc2": (2048, 9216, 0),
    "head": (135168, 2048, 0),
}


def quantize_rowwise(w):
    s = (w.float().abs().amax(dim=1) / 448.0).clamp(min=1e-12)
    q = (w.float() / s[:, None]).to(torch.float8_e4m3fn)
    return q.view(torch.uint8).contiguous(), s.contiguous()


def timed(fn, iters):
    """Call fn(i) with i = 0, 1, … (rotating through the copies)."""
    for i in range(10):
        fn(i)
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    a.record()
    for i in range(iters):
        fn(i)
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000  # µs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("so")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.ops.load_library(args.so)
    g = torch.Generator(device="cuda").manual_seed(0)
    res = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "shapes": {}}
    for name, (n, k, act) in SHAPES.items():
        w = torch.randn(n, k, device="cuda", generator=g, dtype=torch.float32).to(torch.bfloat16)
        x = torch.randn(1, k, device="cuda", generator=g, dtype=torch.float32).to(torch.bfloat16)
        q, s = quantize_rowwise(w)
        y = torch.ops.a2e.fp8_gemv(x, q, s, act, 1)
        nc = max(1, -(-512 * 2**20 // (n * k)))  # number of copies for the FP8 weights to exceed 512 MB in total
        qs = [q.clone() for _ in range(nc)]
        ws = [w.clone() for _ in range(max(1, nc // 2))]  # BF16 is twice the size, so half as many
        ref = F.linear(x.float(), q.view(torch.float8_e4m3fn).float() * s[:, None])
        if act == 1:
            ref = torch.relu(ref) ** 2
        rel = ((y.float() - ref).norm() / ref.norm()).item()
        t_fp8 = timed(lambda i: torch.ops.a2e.fp8_gemv(x, qs[i % nc], s, act, 1), args.iters)
        t_bf16 = timed(lambda i: F.linear(x, ws[i % len(ws)]), args.iters)
        cnt = torch.zeros(1, dtype=torch.int32, device="cuda")
        blocks = torch.cuda.get_device_properties(0).multi_processor_count * 8
        t_read = timed(lambda i: torch.ops.a2e.read_only(qs[i % nc].view(-1), cnt, blocks), args.iters)
        del qs, ws
        torch.cuda.empty_cache()
        r = {"N": n, "K": k, "act": act, "copies": nc, "rel_err_vs_fp32_dequant": rel, "fp8_us": round(t_fp8, 2),
             "bf16_linear_us": round(t_bf16, 2), "read_only_us": round(t_read, 2),
             "fp8_GBps": round(n * k / t_fp8 / 1e3, 1), "bf16_GBps": round(2 * n * k / t_bf16 / 1e3, 1)}
        res["shapes"][name] = r
        print(name, r, flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

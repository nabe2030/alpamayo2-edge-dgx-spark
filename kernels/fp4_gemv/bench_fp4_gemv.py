#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# bench_fp4_gemv.py — how close the FP4 / FP8 / BF16 decode GEMVs get to the GB10 memory-bandwidth ceiling.
#
# 1. ceiling   : a read-only reduction over a 1 GiB buffer (torch.sum of int32 view) and a device copy
# 2. per shape : each Alpamayo 2 Edge decode projection (qkv, o, fc1+relu2, fc2, lm_head), weights rotated
#                over >= 1 GiB of copies so nothing stays in L2 (24 MB on GB10), timed with CUDA events
# 3. body graph: the 28 layers x (qkv, o, fc1, fc2) GEMV chain (distinct real-size weights per layer,
#                random values) captured in one CUDA graph, i.e. the weight-reading part of one decode token
#                without attention / norms. Reported as ms per token and effective GB/s.
# Needs no data and no model (random weights of the real shapes). Usage (after `source env.sh` and building both kernels):
#   $PY bench_fp4_gemv.py --out fp4_gemv_bench.json      (A2E_FP4_SO / A2E_FP8_SO point to the built .so files)
# Our result on GB10 is results/fp4/fp4_gemv_bench.json.
import argparse
import json
import os
import time

import torch
import torch.nn.functional as Fn

import a2e_fp8 as Q
import a2e_fp4 as F

SHAPES = {"qkv": (4096, 2048, 0), "o": (2048, 2048, 0), "fc1": (9216, 2048, 1), "fc2": (2048, 9216, 0),
          "head": (135168, 2048, 0)}


def ev_time(fn, iters):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn()
    torch.cuda.synchronize()
    s.record()
    for i in range(iters):
        fn(i)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def ceiling():
    n = 1 << 30
    buf = torch.empty(n, dtype=torch.uint8, device="cuda").random_(0, 255)
    v = buf.view(torch.int32)
    dst = torch.empty_like(buf)
    t_sum = ev_time(lambda i=0: v.sum(), 20)
    t_cp = ev_time(lambda i=0: dst.copy_(buf), 20)
    return {"read_sum_GBps": n / t_sum / 1e6, "copy_GBps": 2 * n / t_cp / 1e6}


def make(kind, N, K, act):
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    if kind == "fp4":
        qw, bs, gs = F.quantize_nvfp4(w)
        nb = qw.numel() + bs.numel() + gs.numel() * 4
        return (lambda x: torch.ops.a2e4.fp4_gemv(x, qw, bs, gs, act)), nb
    if kind == "fp8":
        q, s = Q.quantize_rowwise(w)
        return (lambda x: torch.ops.a2e.fp8_gemv(x, q, s, act, 1)), q.numel() + s.numel() * 4
    if act:
        return (lambda x: Fn.relu(Fn.linear(x, w)) ** 2), w.numel() * 2
    return (lambda x: Fn.linear(x, w)), w.numel() * 2


def per_shape(kinds):
    out = []
    for name, (N, K, act) in SHAPES.items():
        x = torch.randn(1, 1, K, device="cuda", dtype=torch.bfloat16)
        for kind in kinds:
            f0, nb = make(kind, N, K, act)
            copies = max(2, min(64, (1 << 30) // nb + 1)) if name != "head" else 4
            fs = [f0] + [make(kind, N, K, act)[0] for _ in range(copies - 1)]
            t = ev_time(lambda i=0: fs[i % copies](x), 200 if name != "head" else 40)
            out.append({"shape": name, "N": N, "K": K, "kind": kind, "us": t * 1000, "bytes": nb,
                        "GBps": nb / t / 1e6, "copies": copies})
            print(f"{name:5s} {kind}: {t*1000:8.1f} us  {nb/t/1e6:6.1f} GB/s", flush=True)
            del fs
            torch.cuda.empty_cache()
    return out


def body_graph(kind, layers=28):
    seq = []
    nbytes = 0
    for _ in range(layers):
        for name in ("qkv", "o", "fc1", "fc2"):
            N, K, act = SHAPES[name]
            f, nb = make(kind, N, K, act)
            seq.append((f, K))
            nbytes += nb
    xs = {K: torch.randn(1, 1, K, device="cuda", dtype=torch.bfloat16) for K in (2048, 9216)}

    def run():
        y = None
        for f, K in seq:
            y = f(xs[K])
        return y

    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    t = ev_time(lambda i=0: g.replay(), 50)
    return {"kind": kind, "ms_per_token": t, "bytes": nbytes, "GBps": nbytes / t / 1e6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="fp4,fp8,bf16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    F.load_ops()
    Q.load_ops()
    torch.manual_seed(0)
    res = {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "device": torch.cuda.get_device_name(), "torch": torch.__version__}
    res["ceiling"] = ceiling()
    print(res["ceiling"], flush=True)
    kinds = a.kinds.split(",")
    res["per_shape"] = per_shape(kinds)
    res["body_graph"] = []
    for k in kinds:
        r = body_graph(k)
        print(f"body graph {k}: {r['ms_per_token']:.3f} ms/token {r['GBps']:.1f} GB/s", flush=True)
        res["body_graph"].append(r)
        torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

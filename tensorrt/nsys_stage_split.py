#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
"""Split a TRT Edge-LLM nsys capture into stages per request.

Every TensorRT `ExecutionContext::enqueueV3` NVTX range is one engine call. Kernels are attributed to
the enqueue whose CPU range contains their launch (runtime API call). Per request the call order is
visual engine -> LLM prefill -> one LLM call per further generated token. For each call we report its
GPU window (first kernel start .. last kernel end). Kernels launched outside any enqueue are the
runtime's own CUDA work (image normalize/patchify, rope tables, embedding lookup, sampling, ...).
"""
import argparse, bisect, sqlite3, statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--per-req", type=int, default=19, help="engine calls per request (1 ViT + 1 prefill + n-1 decode)")
    ap.add_argument("--skip", type=int, default=0, help="drop this many leading engine calls (e.g. a warm-up enqueue)")
    a = ap.parse_args()
    c = sqlite3.connect(a.sqlite)
    enq = c.execute("select e.start, e.end from NVTX_EVENTS e left join StringIds s on e.textId = s.id "
                    "where coalesce(e.text, s.value) = 'ExecutionContext::enqueueV3' and e.end is not null "
                    "order by e.start").fetchall()
    rt = dict(c.execute("select correlationId, start from CUPTI_ACTIVITY_KIND_RUNTIME"))
    ks = c.execute("select start, end, correlationId from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    starts = [e[0] for e in enq]
    win = [[None, None, 0] for _ in enq]
    outside = []
    for st, en, cid in ks:
        t = rt.get(cid)
        i = bisect.bisect_right(starts, t) - 1 if t is not None else -1
        if i >= 0 and enq[i][0] <= t <= enq[i][1]:
            w = win[i]
            w[0] = st if w[0] is None else min(w[0], st)
            w[1] = en if w[1] is None else max(w[1], en)
            w[2] += en - st
        else:
            outside.append((st, en))
    calls = [((w[1] - w[0]) / 1e6 if w[0] else 0.0, w[2] / 1e6, w[0], w[1]) for w in win][a.skip:]
    n = a.per_req
    reqs = [calls[i:i + n] for i in range(0, len(calls) - n + 1, n)]
    outside.sort()
    os_ = [o[0] for o in outside]
    for k, r in enumerate(reqs):
        vit, pre, dec = r[0], r[1], r[2:]
        # runtime-own kernels between the start of this request's first call window and its last
        t0, t1 = r[0][2], r[-1][3]
        own = sum((e - s) for s, e in outside[bisect.bisect_left(os_, t0 - 200_000_000):bisect.bisect_right(os_, t1)]
                  if t0 - 200_000_000 <= s <= t1) / 1e6
        gap_vit_pre = (pre[2] - vit[3]) / 1e6
        print(f"req{k}: vit {vit[0]:.1f} ms | vit->prefill gap {gap_vit_pre:.1f} | prefill {pre[0]:.1f} | "
              f"decode calls {len(dec)}: median {statistics.median([d[0] for d in dec]):.2f} ms, "
              f"sum {sum(d[0] for d in dec):.1f}, first->last span {(dec[-1][3] - dec[0][2]) / 1e6:.1f} | "
              f"GPU span vit start..last decode end {(t1 - t0) / 1e6:.1f} | runtime-own kernels ~{own:.1f}")
    print("total enqueues", len(calls))


if __name__ == "__main__":
    main()

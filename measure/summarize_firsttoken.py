#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# summarize_firsttoken.py <firsttoken.jsonl> <out.md> — summary of av2_firsttoken.py.
#   Per variant: mean probability of "go"-type, "stop"-type and "Dec" first tokens, and the difference to base paired by
#   scene (95% interval by resampling scenes). Tokens are grouped as in summarize_stopgo.py.
#   Set A = scenes where the recorded car stays stopped, set B = scenes with a "stop"-type reasoning (av2_stopgo.py plan).
import json
import sys

import numpy as np

GO = {"Pro", "Turn", "Acc", "Adv", "Enter", "Ste", "Change", "Move", "Execute", "Take", "Complete", "Follow",
      "Pull", "Finish", "Appro", "Merge", "Go", "Res"}
STOP = {"Rem", "Wait", "Stop", "Y", "Hold", "Stay", "Come", "Bring"}


def mass(top, S):
    return sum(p for t, _, p in top if t.lstrip("Ġ") in S)


def boot(x, B=2000):
    rng = np.random.default_rng(0)
    x = np.asarray(x)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main(inp, out):
    T = [json.loads(l) for l in open(inp)]
    vs = list(T[0]["top"])
    L = [f"# Probability of the first reasoning token (variants, {len(T)} scenes)\n",
         "| Set | Scenes | Variant | go-type | stop-type | Dec | go-type difference (variant - BF16) [95%] | up / down |", "|---|---|---|---|---|---|---|---|"]
    for sname, sel in (("A", lambda t: t["tier"] == "A"), ("B", lambda t: t["tier"] == "B"), ("A+B", lambda t: True)):
        ts = [t for t in T if sel(t)]
        if not ts:
            continue
        for v in vs:
            g = [mass(t["top"][v], GO) for t in ts]
            st = [mass(t["top"][v], STOP) for t in ts]
            de = [mass(t["top"][v], {"Dec"}) for t in ts]
            if v == "base":
                d = "—"
                ud = "—"
            else:
                dd = [mass(t["top"][v], GO) - mass(t["top"]["base"], GO) for t in ts]
                lo, hi = boot(dd)
                d = f"{100 * np.mean(dd):+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] pt"
                ud = f"{sum(x > 0 for x in dd)} / {sum(x < 0 for x in dd)}"
            L.append(f"| {sname} | {len(ts)} | {v} | {100 * np.mean(g):.1f}% | {100 * np.mean(st):.1f}% | {100 * np.mean(de):.1f}% | {d} | {ud} |")
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(*sys.argv[1:3])

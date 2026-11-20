#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_wide_cases.py <wide.jsonl> <out.md> — look individually at the scenes with large differences
#   1. Top scenes: per seed, BF16 and FP8 reasoning, ADE vs. the recorded drive, same-seed trajectory difference
#   2. Overall: for (scene, seed) where the FP8 CoT diverged from BF16, whether FP8's ADE vs. the recorded drive falls within the BF16 5-seed range (min–max)
#      (the probability that a random 6th sample falls within the range of 5 is 4/6 ≈ 67%)
import json
import math
import sys

def ade(a, b):
    return sum(math.dist(p[:2], r[:2]) for p, r in zip(a, b)) / len(a)

def ade3(a, b):
    return sum(math.dist(p, r) for p, r in zip(a, b)) / len(a)

recs = [json.loads(l) for l in open(sys.argv[1]) if '"kind": "sample"' in l]
L = ["# Scenes with large differences (AV2, wide_ft, seed 43–47)", ""]
inside = {"fp8body": [0, 0], "fp8all": [0, 0]}
rows = []
for r in recs:
    gt, b = r["gt_xyz"], r["sampled"]["base"]
    bad = {s: ade(b[s]["xyz"], gt) for s in b}
    lo, hi = min(bad.values()), max(bad.values())
    seedw = sum(ade3(b[i]["xyz"], b[j]["xyz"]) for i in b for j in b if i < j) / 10
    for v in ("fp8body", "fp8all"):
        d = r["sampled"][v]
        same = sum(ade3(d[s]["xyz"], b[s]["xyz"]) for s in b) / len(b)
        rows.append((same / max(seedw, 1e-3), v, r))
        for s in b:
            if d[s]["gen_ids"] != b[s]["gen_ids"]:
                inside[v][1] += 1
                inside[v][0] += lo <= ade(d[s]["xyz"], gt) <= hi
L += ["## When the FP8 reasoning diverges from BF16: share whose error vs. the recorded drive is within the BF16 5-seed range", ""]
for v, (a, n) in inside.items():
    L.append(f"- {v}: {a}/{n} = {100 * a / n:.0f}% (about 67% if random)")
L += ["", "## Top scenes (same-seed trajectory difference ÷ BF16 seed spread)", ""]
seen = set()
for rel, v, r in sorted(rows, key=lambda x: -x[0])[:8]:
    if (r["sample_id"], v) in seen:
        continue
    seen.add((r["sample_id"], v))
    gt, b, d = r["gt_xyz"], r["sampled"]["base"], r["sampled"][v]
    L += [f"### {r['sample_id']} ({v}, ratio {100 * rel:.0f}%, motion {r['motion']}, {r['city']}, navigation text \"{r['nav_text']}\")", "",
          "| seed | BF16 reasoning | BF16 ADE | " + v + " reasoning | ADE | Same-seed trajectory difference |", "|---|---|---|---|---|---|"]
    for s in sorted(b, key=int):
        same = "same" if d[s]["gen_ids"] == b[s]["gen_ids"] else d[s]["cot"]
        L.append(f"| {s} | {b[s]['cot']} | {ade(b[s]['xyz'], gt):.2f} | {same} | {ade(d[s]['xyz'], gt):.2f} | {ade3(d[s]['xyz'], b[s]['xyz']):.2f} |")
    L.append("")
open(sys.argv[2], "w").write("\n".join(L) + "\n")
print("\n".join(L[:6]))

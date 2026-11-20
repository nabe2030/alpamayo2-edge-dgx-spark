#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_wide_cases.py <wide.jsonl> <out.md> — 差が大きかった場面を個別に見る
#   1. 上位の場面: seed ごとに、BF16 と FP8 の判断理由・記録との ADE・同じ seed の軌跡の差
#   2. 全体: FP8 の CoT が BF16 と分かれた (場面, seed) で、FP8 の記録との ADE が BF16 の 5 seed の幅(最小〜最大)に入るか
#      (5 seed の幅の中に、ランダムな 6 本目が入る確率は 4/6 ≈ 67%)
import json
import math
import sys

def ade(a, b):
    return sum(math.dist(p[:2], r[:2]) for p, r in zip(a, b)) / len(a)

def ade3(a, b):
    return sum(math.dist(p, r) for p, r in zip(a, b)) / len(a)

recs = [json.loads(l) for l in open(sys.argv[1]) if '"kind": "sample"' in l]
L = ["# 差が大きかった場面(AV2、wide_ft、seed 43〜47)", ""]
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
L += ["## FP8 の判断理由が BF16 と分かれたとき、記録との誤差が BF16 の 5 seed の幅に入る割合", ""]
for v, (a, n) in inside.items():
    L.append(f"- {v}: {a}/{n} = {100 * a / n:.0f}%(ランダムなら約 67%)")
L += ["", "## 上位の場面(同じ seed の軌跡差 ÷ BF16 の seed の幅)", ""]
seen = set()
for rel, v, r in sorted(rows, key=lambda x: -x[0])[:8]:
    if (r["sample_id"], v) in seen:
        continue
    seen.add((r["sample_id"], v))
    gt, b, d = r["gt_xyz"], r["sampled"]["base"], r["sampled"][v]
    L += [f"### {r['sample_id']}({v}、比 {100 * rel:.0f}%、走り方 {r['motion']}、{r['city']}、ナビ文 \"{r['nav_text']}\")", "",
          "| seed | BF16 の判断理由 | BF16 ADE | " + v + " の判断理由 | ADE | 同じ seed の軌跡差 |", "|---|---|---|---|---|---|"]
    for s in sorted(b, key=int):
        same = "同じ" if d[s]["gen_ids"] == b[s]["gen_ids"] else d[s]["cot"]
        L.append(f"| {s} | {b[s]['cot']} | {ade(b[s]['xyz'], gt):.2f} | {same} | {ade(d[s]['xyz'], gt):.2f} | {ade3(d[s]['xyz'], b[s]['xyz']):.2f} |")
    L.append("")
open(sys.argv[2], "w").write("\n".join(L) + "\n")
print("\n".join(L[:6]))

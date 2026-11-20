#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# summarize_multi.py <dir> <out.md> <out.json>
# ・§3.3 の連続の測定(複数のログ)の集計。<dir> の *.jsonl(M10_* / M1_* / E10_* / E1_*)を読む。
#   - 条件ごと: 計測の歩(phase = stream)の中央値・p90・最大・1 秒を超えた割合、最も遅い 5 回(ログ・t0・トークン数・段の内訳)
#   - ログごと: 中央値・最大・1 秒を超えた割合・トークン数の中央値と最大
#   - 遅い回の原因: 時間 = a + b × トークン数 の当てはめ(条件ごと)と、当てはめから最も上に外れた回(トークン数以外の要因)
#   - 描き直し込み(E*): 描き直し・推論・合計それぞれの中央値・p90・最大・1 秒を超えた割合
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np


def q(xs, p):
    return float(np.percentile(xs, p * 100)) if len(xs) else float("nan")


def f(x, nd=0):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def stages(r):
    s = r["stages_ms"]
    return f"vit {f(s.get('vision'))} / prefill {f(s.get('prefill_llm'))} / decode {f(s.get('decode'))} / expert {f(s.get('expert'))}"


def main(d, out_md, out_json):
    d = Path(d)
    J, L = {"conditions": {}}, []
    files = sorted(d.glob("*.jsonl"))
    for fp in files:
        name = fp.stem
        rows = [json.loads(l) for l in fp.read_text().splitlines()]
        rs = [r for r in rows if r.get("kind") == "run" and r.get("phase") == "stream"]
        if not rs:
            continue
        done = any(r.get("kind") == "summary" for r in rows)
        key = "total_ms" if "total_ms" in rs[0] else "e2e_ms"
        e = [r[key] for r in rs]
        tok = [r["cot_tokens"] for r in rs]
        a, b = np.polyfit(tok, [r["e2e_ms"] for r in rs], 1)[::-1] if len(set(tok)) > 1 else (float("nan"), float("nan"))
        resid = [(r["e2e_ms"] - (a + b * r["cot_tokens"]), r) for r in rs]
        resid.sort(key=lambda x: -x[0])
        C = {"complete": done, "n": len(rs), "n_logs": len({r["log_id"] for r in rs}), "metric": key,
             "median": st.median(e), "p90": q(e, .9), "max": max(e), "over_1s": sum(x > 1000 for x in e) / len(e),
             "tokens_median": st.median(tok), "tokens_max": max(tok), "fit_intercept_ms": a, "fit_ms_per_token": b,
             "per_log": {}, "slowest": [], "top_residual": []}
        if key == "total_ms":
            for k in ("render_ms", "e2e_ms"):
                v = [r[k] for r in rs]
                C[k] = {"median": st.median(v), "p90": q(v, .9), "max": max(v), "over_1s": sum(x > 1000 for x in v) / len(v)}
        for lid in dict.fromkeys(r["log_id"] for r in rs):
            g = [r for r in rs if r["log_id"] == lid]
            v = [r[key] for r in g]
            C["per_log"][lid] = {"n": len(g), "median": st.median(v), "max": max(v),
                                 "over_1s": sum(x > 1000 for x in v) / len(v),
                                 "tokens_median": st.median([r["cot_tokens"] for r in g]),
                                 "tokens_max": max(r["cot_tokens"] for r in g)}
        for r in sorted(rs, key=lambda r: -r[key])[:5]:
            C["slowest"].append({"log_id": r["log_id"], "t0_us": r["t0_us"], "pass": r["pass_"], key: r[key],
                                 "e2e_ms": r["e2e_ms"], "render_ms": r.get("render_ms"), "tokens": r["cot_tokens"],
                                 "images_vit": r["images_vit"], "temp_before": r.get("temp_before"),
                                 "stages_ms": r["stages_ms"], "cot": r["cot"]})
        for res_, r in resid[:5]:
            C["top_residual"].append({"log_id": r["log_id"], "t0_us": r["t0_us"], "pass": r["pass_"], "e2e_ms": r["e2e_ms"],
                                      "residual_ms": res_, "tokens": r["cot_tokens"], "images_vit": r["images_vit"],
                                      "stages_ms": r["stages_ms"]})
        J["conditions"][name] = C
    # ---- 表
    L.append("## 条件ごと(計測の歩。1 秒を超えた割合は推論だけ、E* は描き直し込みの合計)\n")
    L.append("| 条件 | 済み | 歩数 | ログ | 中央値 ms | p90 | 最大 | 1 秒超え | トークン 中央値/最大 | 当てはめ ms + ms/トークン |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for n, C in J["conditions"].items():
        L.append(f"| {n} | {'済' if C['complete'] else '途中'} | {C['n']} | {C['n_logs']} | **{f(C['median'])}** | {f(C['p90'])} | "
                 f"{f(C['max'])} | **{100 * C['over_1s']:.1f}%** | {f(C['tokens_median'])} / {C['tokens_max']} | "
                 f"{f(C['fit_intercept_ms'])} + {f(C['fit_ms_per_token'], 2)} |")
    for n, C in J["conditions"].items():
        if C["metric"] == "total_ms":
            L.append(f"\n- {n}: 描き直し 中央値 {f(C['render_ms']['median'], 1)} / p90 {f(C['render_ms']['p90'], 1)} / 最大 "
                     f"{f(C['render_ms']['max'], 1)} ms、推論 中央値 {f(C['e2e_ms']['median'])} / 最大 {f(C['e2e_ms']['max'])} ms、"
                     f"合計 中央値 {f(C['median'])} / 最大 {f(C['max'])} ms(1 秒超え {100 * C['over_1s']:.1f}%)")
    L.append("\n## ログごと(中央値 / 最大 / 1 秒超え、トークン数の中央値/最大)\n")
    names = list(J["conditions"])
    logs = list(dict.fromkeys(l for C in J["conditions"].values() for l in C["per_log"]))
    L.append("| ログ | " + " | ".join(names) + " |")
    L.append("|---|" + "---|" * len(names))
    for lid in logs:
        cells = []
        for n in names:
            p = J["conditions"][n]["per_log"].get(lid)
            cells.append("—" if p is None else f"{f(p['median'])} / {f(p['max'])} / {100 * p['over_1s']:.0f}% ({f(p['tokens_median'])}/{p['tokens_max']})")
        L.append(f"| {lid[:8]} | " + " | ".join(cells) + " |")
    L.append("\n## 最も遅い回(条件ごとに 5 回)\n")
    for n, C in J["conditions"].items():
        L.append(f"- **{n}**")
        for s in C["slowest"]:
            L.append(f"  - {s['log_id'][:8]} t0={s['t0_us'] / 1e6:.1f}s pass{s['pass']}: {f(s[C['metric']])} ms"
                     f"{'(描き直し ' + f(s['render_ms']) + ')' if s.get('render_ms') is not None else ''}、"
                     f"{s['tokens']} トークン、ViT {s['images_vit']} 枚、{s['temp_before']}℃、{stages({'stages_ms': s['stages_ms']})}")
        L.append(f"  - 当てはめから最も上に外れた回(トークン数で説明できない分): " + "; ".join(
            f"{x['log_id'][:8]} t0={x['t0_us'] / 1e6:.1f}s +{f(x['residual_ms'])} ms({x['tokens']} トークン、{stages(x)})"
            for x in C["top_residual"][:3]))
    Path(out_md).write_text("\n".join(L) + "\n")
    Path(out_json).write_text(json.dumps(J, indent=1, ensure_ascii=False, default=float))
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main(*sys.argv[1:4])

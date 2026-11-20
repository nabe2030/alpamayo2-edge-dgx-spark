#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# summarize_stopgo.py <stopgo.jsonl> <out.md> <out.json>
#  の集計。分け方は av2_stopgo_common.py(軌跡: 3.0 s で 2.0 m 以上 = 進む。文: 頭の動詞句)。
#   集合: A = 記録された走行が 3.0 s で 1.0 m 未満しか動かない(止まるべき場面)、B = 判断理由に「止まる」が出た場面(A を除く)
#   1. 変種ごと・集合ごとの「進む」の割合(軌跡/文/両方)。区間は場面ごとに引き直すブートストラップ(B=2000)の 95%
#   2. 同じ (場面, seed) で対にした差(FP8 − BF16)と、食い違いの向き(BF16 止まる & FP8 進む / その逆)
#   3. 文が BF16 で stop、FP8 で go に分かれた対で、どこで分かれたか(単語の位置と語)
#   4. 場面ごと: BF16 では一度も「進む」が出ないのに FP8 で出た場面
import collections
import json
import statistics as stt
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from av2_stopgo_common import cot_class, traj_class  # noqa: E402

VARS = ("base", "fp8body", "fp8all")
NAME = {"base": "BF16", "fp8body": "本体だけ FP8", "fp8all": "本体 + 出力層 FP8"}


def boot(scene_vals, fn, B=2000, seed=0):
    """scene_vals: {scene: [値...]}。場面を引き直して fn(全値) の 95% 区間。"""
    rng = np.random.default_rng(seed)
    keys = list(scene_vals)
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        vals = [v for i in pick for v in scene_vals[keys[i]]]
        out.append(fn(vals))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def pct(x):
    return f"{100 * x:.1f}%"


def main(inp, out_md, out_json):
    recs = [json.loads(l) for l in open(inp)]
    for r in recs:
        for v in VARS:
            o = r["out"][v]
            o["tc"] = traj_class(o["xyz"])
            o["cc"] = cot_class(o["cot"])
    J, L = {"n_records": len(recs), "sets": {}}, []
    sets = {"A": [r for r in recs if r["tier"] == "A"], "B": [r for r in recs if r["tier"] == "B"], "A+B": recs}
    L.append(f"記録 {len(recs)} 件(場面 × seed)。各件で BF16・本体だけ FP8・本体 + 出力層 FP8 を同じ seed で回した。\n")
    for sname, rs in sets.items():
        if not rs:
            continue
        by_scene = collections.defaultdict(list)
        for r in rs:
            by_scene[r["scene"]].append(r)
        S = {"n": len(rs), "n_scenes": len(by_scene), "rates": {}, "paired": {}, "cot_dist": {}}
        L.append(f"## 集合 {sname}(場面 {len(by_scene)}、件数 {len(rs)})\n")
        L.append("| 変種 | 軌跡で進む [95%] | 文で go [95%] | 文 go かつ軌跡で進む [95%] | 文の分類(stop/go/creep/slow/nominal/other) |")
        L.append("|---|---|---|---|---|")
        for v in VARS:
            def rate(key, val, v=v):
                sv = {s: [int(r["out"][v][key] == val) for r in g] for s, g in by_scene.items()}
                m = float(np.mean([x for g in sv.values() for x in g]))
                return m, boot(sv, np.mean)
            tg, cg = rate("tc", "go"), rate("cc", "go")
            sv = {s: [int(r["out"][v]["tc"] == "go" and r["out"][v]["cc"] == "go") for r in g] for s, g in by_scene.items()}
            both = (float(np.mean([x for g in sv.values() for x in g])), boot(sv, np.mean))
            dist = collections.Counter(r["out"][v]["cc"] for r in rs)
            S["rates"][v] = {"traj_go": tg, "cot_go": cg, "both_go": both}
            S["cot_dist"][v] = dict(dist)
            L.append(f"| {NAME[v]} | {pct(tg[0])} [{pct(tg[1][0])}, {pct(tg[1][1])}] | {pct(cg[0])} [{pct(cg[1][0])}, {pct(cg[1][1])}] | "
                     f"{pct(both[0])} [{pct(both[1][0])}, {pct(both[1][1])}] | "
                     + " / ".join(str(dist.get(k, 0)) for k in ("stop", "go", "creep", "slow", "nominal", "other")) + " |")
        L.append("\n対にした差(同じ場面・同じ seed。FP8 − BF16、ポイント)と、食い違いの向き:\n")
        L.append("| 変種 | 軌跡で進む の差 [95%] | BF16 止まる→FP8 進む | BF16 進む→FP8 止まる | 文 go の差 [95%] | 文 stop→go | 文 go→stop |")
        L.append("|---|---|---|---|---|---|---|")
        for v in VARS[1:]:
            res = {}
            for key in ("tc", "cc"):
                sv = {s: [int(r["out"][v][key] == "go") - int(r["out"]["base"][key] == "go") for r in g] for s, g in by_scene.items()}
                m = float(np.mean([x for g in sv.values() for x in g]))
                s2g = sum(r["out"]["base"][key] == "stop" and r["out"][v][key] == "go" for r in rs)
                g2s = sum(r["out"]["base"][key] == "go" and r["out"][v][key] == "stop" for r in rs)
                res[key] = {"diff": m, "ci": boot(sv, np.mean), "stop_to_go": s2g, "go_to_stop": g2s}
            S["paired"][v] = res
            t, c = res["tc"], res["cc"]
            L.append(f"| {NAME[v]} | {100 * t['diff']:+.1f} [{100 * t['ci'][0]:+.1f}, {100 * t['ci'][1]:+.1f}] | {t['stop_to_go']} | {t['go_to_stop']} | "
                     f"{100 * c['diff']:+.1f} [{100 * c['ci'][0]:+.1f}, {100 * c['ci'][1]:+.1f}] | {c['stop_to_go']} | {c['go_to_stop']} |")
        L.append("")
        J["sets"][sname] = S
    # 3. 分かれた位置
    L.append("## 文が BF16 で stop、FP8 で go に分かれた対の、分かれた位置\n")
    J["splits"] = {}
    for v in VARS[1:]:
        sp = []
        for r in recs:
            b, d = r["out"]["base"], r["out"][v]
            if b["cc"] == "stop" and d["cc"] == "go":
                i = next((k for k, (x, y) in enumerate(zip(b["gen_ids"], d["gen_ids"])) if x != y), None)
                wb, wd = b["cot"].split(), d["cot"].split()
                w = next((k for k, (x, y) in enumerate(zip(wb, wd)) if x != y), None)
                sp.append({"scene": r["scene"], "seed": r["seed"], "token_pos": i, "word_pos": w,
                           "base_word": wb[w] if w is not None else None, "fp8_word": wd[w] if w is not None else None,
                           "base": b["cot"], "fp8": d["cot"], "traj_base": b["tc"], "traj_fp8": d["tc"]})
        J["splits"][v] = sp
        pos = collections.Counter(s["token_pos"] for s in sp)
        words = collections.Counter((s["base_word"], s["fp8_word"]) for s in sp)
        L.append(f"- **{NAME[v]}**: {len(sp)} 対。分かれたトークンの位置 {dict(sorted(pos.items(), key=lambda x: (x[0] is None, x[0])))}。"
                 f"語の入れ替わり(上位): " + "、".join(f"{a}→{b} {n}" for (a, b), n in words.most_common(6)))
        for s in sp[:4]:
            L.append(f"  - {s['scene']} s{s['seed']}: 「{s['base'][:90]}」→「{s['fp8'][:90]}」(軌跡 {s['traj_base']}→{s['traj_fp8']})")
    # 4. 場面ごと
    L.append("\n## BF16 では「進む」が一度も出ないのに FP8 で出た場面(軌跡で。文で go の回数も併記)\n")
    by_scene = collections.defaultdict(list)
    for r in recs:
        by_scene[r["scene"]].append(r)
    J["scenes"] = {}
    rows = []
    for s, g in by_scene.items():
        c = {v: sum(r["out"][v]["tc"] == "go" for r in g) for v in VARS}
        cc = {v: sum(r["out"][v]["cc"] == "go" for r in g) for v in VARS}
        J["scenes"][s] = {"tier": g[0]["tier"], "n": len(g), "traj_go": c, "cot_go": cc}
        if c["base"] == 0 and (c["fp8body"] > 0 or c["fp8all"] > 0):
            rows.append((s, g[0]["tier"], len(g), c, cc))
    L.append("| 場面 | 集合 | seed 数 | 軌跡で進む BF16 / 本体 / 本体+出力層 | 文で go BF16 / 本体 / 本体+出力層 |")
    L.append("|---|---|---|---|---|")
    for s, t, n, c, cc in rows:
        L.append(f"| {s} | {t} | {n} | {c['base']} / {c['fp8body']} / {c['fp8all']} | {cc['base']} / {cc['fp8body']} / {cc['fp8all']} |")
    rows2 = [(s, x) for s, x in J["scenes"].items() if x["traj_go"]["base"] > 0 and x["traj_go"]["fp8body"] == 0 and x["traj_go"]["fp8all"] == 0]
    L.append(f"\n(逆向き: BF16 だけで「進む」が出た場面は {len(rows2)}。)")
    # 5. 1 トークン目の分布(teacher forcing、乱数なし)
    tfp = Path(inp).with_name(Path(inp).stem + "_tf.jsonl")
    if tfp.exists():
        T = [json.loads(l) for l in tfp.read_text().splitlines()]
        # 1 トークン目は単語より細かいトークン(Pro = Proceed、Rem = Remain、Y = Yield など)。出てきたトークンを全部見て分けた
        GO = {"Pro", "Turn", "Acc", "Adv", "Enter", "Ste", "Change", "Move", "Execute", "Take", "Complete", "Follow",
              "Pull", "Finish", "Appro", "Merge", "Go", "Res"}
        STOP = {"Rem", "Wait", "Stop", "Y", "Hold", "Stay", "Come", "Bring"}
        CREEP = {"Cre"}

        def mass(top, S_):
            return sum(p_ for t, _, p_ in top if t.lstrip("Ġ") in S_)
        L.append("\n## 1 トークン目の確率(teacher forcing。BF16 の最初の seed の判断理由を流し、位置 0 の分布を温度 0.6 で)\n")
        L.append("「進む」系の頭のトークン: " + "、".join(sorted(GO)) + "。「止まる」系: " + "、".join(sorted(STOP)) +
                 "。Dec(Decelerate)は止まる/ゆるめるの両方に使われ、Cre(Creep)はゆっくり前に出るので、どちらにも入れない。"
                 "上位 30 トークンの中の確率の和。\n")
        L.append("| 集合 | 場面 | 変種 | 進む系 平均 | 止まる系 平均 | Decelerate 平均 | 進む系の差(FP8 − BF16)[95%] |")
        L.append("|---|---|---|---|---|---|---|")
        J["first_token"] = {}
        for sname, sel in (("A", lambda t: t["tier"] == "A"), ("B", lambda t: t["tier"] == "B"), ("A+B", lambda t: True)):
            ts = [t for t in T if sel(t)]
            if not ts:
                continue
            for v in VARS:
                g = [mass(t["top"][v], GO) for t in ts]
                st_ = [mass(t["top"][v], STOP) for t in ts]
                de = [mass(t["top"][v], {"Dec"}) for t in ts]
                if v == "base":
                    diff = "—"
                    dj = None
                else:
                    dv = {t["scene"]: [mass(t["top"][v], GO) - mass(t["top"]["base"], GO)] for t in ts}
                    m = float(np.mean([x[0] for x in dv.values()]))
                    ci = boot(dv, np.mean)
                    diff = f"{100 * m:+.2f} [{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}] pt"
                    dj = {"diff": m, "ci": ci, "n_up": sum(x[0] > 0 for x in dv.values()), "n_down": sum(x[0] < 0 for x in dv.values())}
                    diff += f"(上がった場面 {dj['n_up']} / 下がった場面 {dj['n_down']})"
                J["first_token"].setdefault(sname, {})[v] = {"go": float(np.mean(g)), "stop": float(np.mean(st_)),
                                                             "decel": float(np.mean(de)), "diff_go": dj}
                L.append(f"| {sname} | {len(ts)} | {NAME[v]} | {pct(np.mean(g))} | {pct(np.mean(st_))} | {pct(np.mean(de))} | {diff} |")
        big = sorted(T, key=lambda t: -(mass(t["top"]["fp8all"], GO) - mass(t["top"]["base"], GO)))[:5]
        L.append("\n進む系の確率が最も上がった場面(本体 + 出力層 FP8): " + "; ".join(
            f"{t['scene']} {pct(mass(t['top']['base'], GO))}→{pct(mass(t['top']['fp8all'], GO))}" for t in big))
    Path(out_md).write_text("\n".join(L) + "\n")
    Path(out_json).write_text(json.dumps(J, indent=1, ensure_ascii=False))
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main(*sys.argv[1:4])

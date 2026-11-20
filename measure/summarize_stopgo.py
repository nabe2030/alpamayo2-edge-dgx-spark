#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# summarize_stopgo.py <stopgo.jsonl> <out.md> <out.json>
#  summary. Classification is from av2_stopgo_common.py (trajectory: ≥ 2.0 m in 3.0 s = go. text: leading verb phrase).
#   Sets: A = the recorded drive moves less than 1.0 m in 3.0 s (stopped scenes), B = scenes where the reasoning says "stop" (excluding A)
#   1. Share of "go" per variant and set (trajectory / text / both). Intervals: 95% from a scene-level bootstrap (B=2000)
#   2. Paired difference on the same (scene, seed) (FP8 − BF16) and direction of disagreement (BF16 stop & FP8 go / the reverse)
#   3. For pairs whose text is stop in BF16 and go in FP8: where they diverged (word position and word)
#   4. Per scene: scenes where BF16 never produced "go" but FP8 did
import collections
import json
import statistics as stt
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from av2_stopgo_common import cot_class, traj_class  # noqa: E402

VARS = ("base", "fp8body", "fp8all")
NAME = {"base": "BF16", "fp8body": "FP8 body only", "fp8all": "FP8 body + output layer"}


def boot(scene_vals, fn, B=2000, seed=0):
    """scene_vals: {scene: [values...]}. 95% interval of fn(all values) by resampling scenes."""
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
    L.append(f"{len(recs)} records (scene × seed). Each record ran BF16, FP8 body only and FP8 body + output layer with the same seed.\n")
    for sname, rs in sets.items():
        if not rs:
            continue
        by_scene = collections.defaultdict(list)
        for r in rs:
            by_scene[r["scene"]].append(r)
        S = {"n": len(rs), "n_scenes": len(by_scene), "rates": {}, "paired": {}, "cot_dist": {}}
        L.append(f"## Set {sname} ({len(by_scene)} scenes, {len(rs)} records)\n")
        L.append("| Variant | Trajectory go [95%] | Text go [95%] | Text go and trajectory go [95%] | Text class (stop/go/creep/slow/nominal/other) |")
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
        L.append("\nPaired difference (same scene and seed; FP8 − BF16, points) and direction of disagreement:\n")
        L.append("| Variant | Trajectory go difference [95%] | BF16 stop→FP8 go | BF16 go→FP8 stop | Text go difference [95%] | Text stop→go | Text go→stop |")
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
    # 3. divergence position
    L.append("## Divergence position for pairs whose text is stop in BF16 and go in FP8\n")
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
        L.append(f"- **{NAME[v]}**: {len(sp)} pairs. Divergent token position {dict(sorted(pos.items(), key=lambda x: (x[0] is None, x[0])))}. "
                 f"Word swaps (top): " + ", ".join(f"{a}→{b} {n}" for (a, b), n in words.most_common(6)))
        for s in sp[:4]:
            L.append(f"  - {s['scene']} s{s['seed']}: \"{s['base'][:90]}\" → \"{s['fp8'][:90]}\" (trajectory {s['traj_base']}→{s['traj_fp8']})")
    # 4. per scene
    L.append("\n## Scenes where BF16 never produced \"go\" but FP8 did (by trajectory; text go counts also shown)\n")
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
    L.append("| Scene | Set | Seeds | Trajectory go BF16 / body / body+output layer | Text go BF16 / body / body+output layer |")
    L.append("|---|---|---|---|---|")
    for s, t, n, c, cc in rows:
        L.append(f"| {s} | {t} | {n} | {c['base']} / {c['fp8body']} / {c['fp8all']} | {cc['base']} / {cc['fp8body']} / {cc['fp8all']} |")
    rows2 = [(s, x) for s, x in J["scenes"].items() if x["traj_go"]["base"] > 0 and x["traj_go"]["fp8body"] == 0 and x["traj_go"]["fp8all"] == 0]
    L.append(f"\n(Reverse direction: scenes where only BF16 produced \"go\": {len(rows2)}.)")
    # 5. first-token distribution (teacher forcing, no sampling)
    tfp = Path(inp).with_name(Path(inp).stem + "_tf.jsonl")
    if tfp.exists():
        T = [json.loads(l) for l in tfp.read_text().splitlines()]
        # The first token is a sub-word token (Pro = Proceed, Rem = Remain, Y = Yield, etc.). Classified by inspecting every token that appeared
        GO = {"Pro", "Turn", "Acc", "Adv", "Enter", "Ste", "Change", "Move", "Execute", "Take", "Complete", "Follow",
              "Pull", "Finish", "Appro", "Merge", "Go", "Res"}
        STOP = {"Rem", "Wait", "Stop", "Y", "Hold", "Stay", "Come", "Bring"}
        CREEP = {"Cre"}

        def mass(top, S_):
            return sum(p_ for t, _, p_ in top if t.lstrip("Ġ") in S_)
        L.append("\n## First-token probability (teacher forcing: feed the reasoning of the first BF16 seed, distribution at position 0 at temperature 0.6)\n")
        L.append("\"Go\" leading tokens: " + ", ".join(sorted(GO)) + ". \"Stop\" tokens: " + ", ".join(sorted(STOP)) +
                 ". Dec (Decelerate) is used both for stopping and slowing down, and Cre (Creep) means moving forward slowly, so neither is counted in either group. "
                 "Sum of probabilities within the top 30 tokens.\n")
        L.append("| Set | Scenes | Variant | Go mean | Stop mean | Decelerate mean | Go difference (FP8 − BF16) [95%] |")
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
                    diff += f" (scenes up {dj['n_up']} / scenes down {dj['n_down']})"
                J["first_token"].setdefault(sname, {})[v] = {"go": float(np.mean(g)), "stop": float(np.mean(st_)),
                                                             "decel": float(np.mean(de)), "diff_go": dj}
                L.append(f"| {sname} | {len(ts)} | {NAME[v]} | {pct(np.mean(g))} | {pct(np.mean(st_))} | {pct(np.mean(de))} | {diff} |")
        big = sorted(T, key=lambda t: -(mass(t["top"]["fp8all"], GO) - mass(t["top"]["base"], GO)))[:5]
        L.append("\nScenes with the largest rise in go probability (FP8 body + output layer): " + "; ".join(
            f"{t['scene']} {pct(mass(t['top']['base'], GO))}→{pct(mass(t['top']['fp8all'], GO))}" for t in big))
    Path(out_md).write_text("\n".join(L) + "\n")
    Path(out_json).write_text(json.dumps(J, indent=1, ensure_ascii=False))
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main(*sys.argv[1:4])

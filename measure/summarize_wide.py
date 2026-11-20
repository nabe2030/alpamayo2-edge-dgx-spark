#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# summarize_wide.py <wide.jsonl> <scenes.csv> <out_prefix> [--ref-1clip <summary_fp8.json>]
#   Summarizes the records of wide_eval.py. Output: <out_prefix>.json (numbers) and <out_prefix>.md (tables).
#
# Yardstick: the difference between BF16 (base) runs that differ only in seed (seed pairs i<j). FP8 variants are compared with base at the same scene and seed.
# Confidence intervals: t0s from the same clip are similar, so 95% intervals come from a clip-level bootstrap (2,000 resamples, seed 0).
# AV2: a "clip" here is an AV2 log (clip_id = log_id). The 2 scenes of the same log are resampled together.
import argparse
import itertools
import json
import math
from collections import defaultdict

import numpy as np
import pandas as pd

B = 2000


def ade(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a[:, :2] - b[:, :2], axis=1).mean()) if a.shape[1] >= 2 else float("nan")


def ade3(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b, axis=1).mean())


def fde(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a[-1] - b[-1]))


def behavior(gt):
    """Roughly classify the behavior from the recorded future trajectory (ego frame at t0, x forward, y left)."""
    if gt is None:
        return "unknown"
    g = np.asarray(gt)
    dist = float(np.linalg.norm(g[-1, :2]))
    if dist < 3.0:
        return "stop"
    head = math.degrees(math.atan2(g[-1, 1] - g[-6, 1], g[-1, 0] - g[-6, 0])) if dist > 5 else 0.0
    if head > 30:
        return "turn_left"
    if head < -30:
        return "turn_right"
    return "straight"


def boot(df, col, stat=np.mean, by="clip_id", seed=0):
    """95% interval of stat(df[col]) by clip-level bootstrap."""
    df = df[[by, col]].dropna()
    if df.empty:
        return [None, None, None]
    groups = [g[col].to_numpy() for _, g in df.groupby(by)]
    rng = np.random.default_rng(seed)
    est = float(stat(df[col].to_numpy()))
    vals = []
    for _ in range(B):
        idx = rng.integers(0, len(groups), len(groups))
        vals.append(stat(np.concatenate([groups[i] for i in idx])))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [est, float(lo), float(hi)]


def boot_ratio(df, num, den, by="clip_id", seed=0):
    """95% interval of mean(num) / mean(den) (clip level)."""
    df = df[[by, num, den]].dropna()
    groups = [(g[num].to_numpy(), g[den].to_numpy()) for _, g in df.groupby(by)]
    rng = np.random.default_rng(seed)
    est = df[num].mean() / df[den].mean()
    vals = []
    for _ in range(B):
        idx = rng.integers(0, len(groups), len(groups))
        n = np.concatenate([groups[i][0] for i in idx])
        d = np.concatenate([groups[i][1] for i in idx])
        vals.append(n.mean() / d.mean())
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(est), float(lo), float(hi)]


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def fmt(ci, d=3, pct=False):
    if ci is None or ci[0] is None:
        return "—"
    k = 100 if pct else 1
    return f"{ci[0]*k:.{d}f} [{ci[1]*k:.{d}f}, {ci[2]*k:.{d}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("scenes")
    ap.add_argument("out")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.jsonl)]
    meta = next(r for r in recs if r["kind"] == "meta")
    samples = [r for r in recs if r["kind"] == "sample"]
    scenes = pd.read_csv(a.scenes)  # AV2: the categories are in the records (sample), so this is not used here
    variants = meta["variants"]
    others = variants[1:]
    seeds = sorted(samples[0]["sampled"]["base"].keys(), key=int)
    tfv = meta["tf_variants"]

    rows, speed = [], []
    for r in samples:
        base = r["sampled"]["base"]
        gt = r["gt_xyz"]
        row = {"sample_id": r["sample_id"], "clip_id": r["clip_id"], "nav_text": r["nav_text"],
               # AV2: motion (category from scene selection), city, split. nav_group is the maneuver in the GT navigation text
               "motion": r.get("motion"), "city": r.get("city"), "split": r.get("split"),
               "nav_group": (r["nav_text"].split(" in ")[0] if r.get("nav_text") else None),
               "behavior": behavior(gt), "temp_max": r.get("temp_max")}
        # spread between BF16 runs (different seeds)
        pairs = list(itertools.combinations(seeds, 2))
        row["bf16_seed_traj"] = float(np.mean([ade3(base[i]["xyz"], base[j]["xyz"]) for i, j in pairs]))
        row["bf16_seed_cot_same"] = float(np.mean([base[i]["gen_ids"] == base[j]["gen_ids"] for i, j in pairs]))
        for v in variants:
            d = r["sampled"][v]
            if gt is not None:
                ades = [ade(d[s]["xyz"], gt) for s in seeds]
                row[f"{v}_ade_gt"] = float(np.mean(ades))
                row[f"{v}_minade_gt"] = float(np.min(ades))
                row[f"{v}_fde_gt"] = float(np.mean([fde(d[s]["xyz"], gt) for s in seeds]))
            row[f"{v}_ntok"] = float(np.mean([d[s]["n_tok"] for s in seeds]))
            row[f"{v}_nominal"] = float(np.mean([("<NOMINAL>" in (d[s]["cot"] or "")) for s in seeds]))
            for s in seeds:
                x = d[s]
                if x.get("new_graphs", 0) == 0:
                    speed.append({"variant": v, "sample_id": r["sample_id"], "clip_id": r["clip_id"], "seed": s,
                                  "e2e_ms": x["e2e_ms"], "decode_ms": x["decode_ms"], "n_tok": x["n_tok"]})
        for v in others:
            d = r["sampled"][v]
            row[f"{v}_same_traj"] = float(np.mean([ade3(d[s]["xyz"], base[s]["xyz"]) for s in seeds]))
            row[f"{v}_same_cot"] = float(np.mean([d[s]["gen_ids"] == base[s]["gen_ids"] for s in seeds]))
            divs = [first_div(d[s]["gen_ids"], base[s]["gen_ids"]) for s in seeds]
            row[f"{v}_div_pos"] = [x for x in divs if x is not None]
            if gt is not None:
                row[f"{v}_d_ade_gt"] = row[f"{v}_ade_gt"] - row["base_ade_gt"]
                row[f"{v}_d_minade_gt"] = row[f"{v}_minade_gt"] - row["base_minade_gt"]
            row[f"{v}_d_ntok"] = row[f"{v}_ntok"] - row["base_ntok"]
            row[f"{v}_d_nominal"] = row[f"{v}_nominal"] - row["base_nominal"]
            row[f"{v}_cot_examples"] = {s: d[s]["cot"] for s in seeds if d[s]["gen_ids"] != base[s]["gen_ids"]}
        row["base_cot"] = {s: base[s]["cot"] for s in seeds}
        # AV2: reference. BF16 with the navigation text always set to "Continue straight"
        if r.get("straight") and gt is not None:
            ss = [s for s in seeds if s in r["straight"]]
            row["straight_ade_gt"] = float(np.mean([ade(r["straight"][s]["xyz"], gt) for s in ss]))
            row["straight_minade_gt"] = float(np.min([ade(r["straight"][s]["xyz"], gt) for s in ss]))
            row["straight_d_ade_gt"] = row["straight_ade_gt"] - float(np.mean([ade(base[s]["xyz"], gt) for s in ss]))
            row["straight_cot_same"] = float(np.mean([r["straight"][s]["gen_ids"] == base[s]["gen_ids"] for s in ss]))
        tf = r["tf"]
        row["tf_noise_alt"] = tf["base_noise_alt_ade"]
        row["tf_ntok"] = tf["n_tok"]
        for v in tfv:
            m = tf["vs"][v]
            row[f"tf_{v}_kl"] = float(np.mean(m["kl_t1"])) if m["kl_t1"] else 0.0
            row[f"tf_{v}_kl06"] = float(np.mean(m["kl_t06"])) if m["kl_t06"] else 0.0
            row[f"tf_{v}_top1"] = float(np.mean(m["top1_same"])) if m["top1_same"] else 1.0
            row[f"tf_{v}_top1_miss"] = int(len(m["top1_same"]) - sum(m["top1_same"]))
            row[f"tf_{v}_traj"] = m["traj_ade_vs_base"]
        rows.append(row)
    df = pd.DataFrame(rows)
    sp = pd.DataFrame(speed)
    res = {"n_samples": len(df), "n_clips": int(df["clip_id"].nunique()), "seeds": seeds, "variants": variants,
           "gt_available": int(df["base_ade_gt"].notna().sum()) if "base_ade_gt" in df else 0}

    # ---- overall
    ov = {"bf16_seed_traj_mean": boot(df, "bf16_seed_traj"), "bf16_seed_traj_median": boot(df, "bf16_seed_traj", np.median),
          "bf16_seed_cot_same": boot(df, "bf16_seed_cot_same"), "tf_noise_alt_mean": boot(df, "tf_noise_alt"),
          "base_ade_gt": boot(df, "base_ade_gt"), "base_minade_gt": boot(df, "base_minade_gt"),
          "base_ntok": boot(df, "base_ntok"), "base_nominal": boot(df, "base_nominal")}
    for v in others:
        ov[v] = {
            "same_seed_traj_mean": boot(df, f"{v}_same_traj"), "same_seed_traj_median": boot(df, f"{v}_same_traj", np.median),
            "ratio_traj_vs_bf16_seed": boot_ratio(df, f"{v}_same_traj", "bf16_seed_traj"),
            "same_seed_cot": boot(df, f"{v}_same_cot"),
            "ade_gt": boot(df, f"{v}_ade_gt"), "d_ade_gt": boot(df, f"{v}_d_ade_gt"),
            "d_minade_gt": boot(df, f"{v}_d_minade_gt"), "d_ntok": boot(df, f"{v}_d_ntok"),
            "d_nominal": boot(df, f"{v}_d_nominal"),
            "div_pos_median": float(np.median([x for l in df[f"{v}_div_pos"] for x in l])) if any(len(l) for l in df[f"{v}_div_pos"]) else None,
        }
    for v in tfv:
        ov[f"tf_{v}"] = {"kl_t1": boot(df, f"tf_{v}_kl"), "kl_t06": boot(df, f"tf_{v}_kl06"),
                         "top1": boot(df, f"tf_{v}_top1"), "top1_miss_total": int(df[f"tf_{v}_top1_miss"].sum()),
                         "n_tok_total": int(df["tf_ntok"].sum()), "traj": boot(df, f"tf_{v}_traj"),
                         "traj_ratio_vs_noise": boot_ratio(df, f"tf_{v}_traj", "tf_noise_alt")}
        if "eager" in tfv and v != "eager":
            ov[f"tf_{v}"]["kl_ratio_vs_eager"] = boot_ratio(df, f"tf_{v}_kl", "tf_eager_kl")
    if "straight_ade_gt" in df:
        g = df[df["straight_ade_gt"].notna()]
        ov["straight"] = {"n": len(g), "ade_gt": boot(g, "straight_ade_gt"), "minade_gt": boot(g, "straight_minade_gt"),
                          "d_ade_gt": boot(g, "straight_d_ade_gt"), "cot_same_as_gt_nav": boot(g, "straight_cot_same"),
                          "by_motion": {k: {"n": len(x), "d_ade_gt": float(x["straight_d_ade_gt"].mean())}
                                        for k, x in g.groupby("motion")}}
    res["overall"] = ov

    # ---- per scene category
    cats = {}
    for col in ("motion", "city", "split", "nav_group", "behavior"):
        cats[col] = {}
        for k, g in df.groupby(col):
            e = {"n": len(g), "n_clips": int(g["clip_id"].nunique()), "bf16_seed_traj": float(g["bf16_seed_traj"].mean()),
                 "bf16_seed_cot_same": float(g["bf16_seed_cot_same"].mean()), "base_ntok": float(g["base_ntok"].mean()),
                 "base_nominal": float(g["base_nominal"].mean())}
            for v in others:
                e[v] = {"same_traj": float(g[f"{v}_same_traj"].mean()),
                        "ratio": float(g[f"{v}_same_traj"].mean() / g["bf16_seed_traj"].mean()),
                        "ratio_ci": boot_ratio(g, f"{v}_same_traj", "bf16_seed_traj") if g["clip_id"].nunique() >= 3 else None,
                        "same_cot": float(g[f"{v}_same_cot"].mean()),
                        "d_ade_gt": float(g[f"{v}_d_ade_gt"].mean()) if f"{v}_d_ade_gt" in g else None,
                        "tf_kl": float(g[f"tf_{v}_kl"].mean()) if f"tf_{v}_kl" in g else None}
            cats[col][str(k)] = e
    res["by_category"] = cats

    # ---- scenes with large differences (FP8 same-seed trajectory difference ÷ BF16 seed spread, and top KL)
    big = {}
    for v in others:
        df[f"{v}_rel"] = df[f"{v}_same_traj"] / df["bf16_seed_traj"].clip(lower=1e-3)
        top = df.sort_values(f"{v}_rel", ascending=False).head(10)
        big[v] = [{"sample_id": t.sample_id, "nav_text": t.nav_text, "motion": t.motion, "city": t.city, "behavior": t.behavior, "same_traj": t[f"{v}_same_traj"], "bf16_seed_traj": t.bf16_seed_traj,
                   "rel": t[f"{v}_rel"], "same_cot": t[f"{v}_same_cot"], "tf_kl": t.get(f"tf_{v}_kl"),
                   "d_ade_gt": t.get(f"{v}_d_ade_gt"), "base_cot": t.base_cot, "var_cot_diff": t[f"{v}_cot_examples"]}
                  for t in (pd.Series(x) for x in top.to_dict("records"))]
        topkl = df.sort_values(f"tf_{v}_kl", ascending=False).head(5) if f"tf_{v}_kl" in df else df.head(0)
        big[f"{v}_kl_top"] = [{"sample_id": t["sample_id"], "nav_text": t["nav_text"], "kl": t[f"tf_{v}_kl"],
                               "eager_kl": t.get("tf_eager_kl"), "top1_miss": t[f"tf_{v}_top1_miss"],
                               "tf_traj": t[f"tf_{v}_traj"], "base_cot": t["base_cot"][seeds[0]]}
                              for t in topkl.to_dict("records")]
    res["largest"] = big

    # ---- speed (single-input measurement; runs that built a new graph are excluded)
    spd = {}
    for v, g in sp.groupby("variant"):
        x, y = g["n_tok"].to_numpy(), g["decode_ms"].to_numpy()
        slope, icpt = np.polyfit(x, y, 1)
        e = g["e2e_ms"].to_numpy()
        spd[v] = {"n_calls": len(g), "ms_per_token_slope": float(slope), "decode_intercept_ms": float(icpt),
                  "e2e_median": float(np.median(e)), "e2e_p90": float(np.percentile(e, 90)), "e2e_max": float(e.max()),
                  "under_1s": float((e < 1000).mean()), "n_tok_median": float(np.median(x)),
                  "n_tok_p90": float(np.percentile(x, 90)), "n_tok_max": int(x.max()),
                  "e2e_at_26tok_fit": None}
        if len(np.unique(x)) > 2:
            s2, i2 = np.polyfit(x, e, 1)
            spd[v]["e2e_fit"] = {"slope": float(s2), "intercept": float(i2), "at_26": float(i2 + 26 * s2)}
            spd[v]["tokens_within_1s_fit"] = float((1000 - i2) / s2)
    hist = np.histogram(sp[sp["variant"] == "base"]["n_tok"], bins=[0, 5, 10, 20, 30, 40, 50, 60, 80, 120, 257])
    res["speed"] = spd
    res["ntok_hist_base"] = {"bins": hist[1].tolist(), "counts": hist[0].tolist()}
    res["excluded_new_graph_calls"] = int(sum(1 for r in samples for v in variants for s in seeds
                                             if r["sampled"][v][s].get("new_graphs", 0) > 0))
    res["temp_max"] = float(df["temp_max"].max())
    json.dump(res, open(a.out + ".json", "w"), indent=1, ensure_ascii=False, default=float)
    df.drop(columns=[c for c in df.columns if c.endswith("_cot_examples") or c == "base_cot" or c.endswith("_div_pos")]) \
      .to_csv(a.out + "_per_sample.csv", index=False)

    # ---- tables (md)
    L = [f"# Summary of the wide-data comparison (AV2, {res['n_samples']} scenes / {res['n_clips']} logs, seed {','.join(seeds)})", ""]
    L += ["## Trajectory (m, mean [log-level bootstrap 95% interval])", "",
          "| Comparison | Value |", "|---|---|",
          f"| BF16 vs BF16, different seed only (mean over seed pairs) | {fmt(ov['bf16_seed_traj_mean'])} |",
          f"| BF16, same CoT, different diffusion noise only (teacher forcing) | {fmt(ov['tf_noise_alt_mean'])} |"]
    for v in others:
        L.append(f"| {v} vs BF16, same seed | {fmt(ov[v]['same_seed_traj_mean'])} ({fmt(ov[v]['ratio_traj_vs_bf16_seed'], 1, True)} % of the BF16 seed spread) |")
    L += ["", "| Error vs. recorded drive | ADE (seed mean) | Difference (variant − BF16) | minADE (5 seeds) difference |", "|---|---|---|---|",
          f"| BF16 | {fmt(ov['base_ade_gt'])} | — | — |"]
    for v in others:
        L.append(f"| {v} | {fmt(ov[v]['ade_gt'])} | {fmt(ov[v]['d_ade_gt'])} | {fmt(ov[v]['d_minade_gt'])} |")
    if "straight" in ov:
        x = ov["straight"]
        L += ["", "### Reference: navigation text always \"Continue straight\" (BF16)", "",
              f"- ADE {fmt(x['ade_gt'])}, difference from GT navigation text {fmt(x['d_ade_gt'])}, minADE {fmt(x['minade_gt'])}, CoT same as with GT navigation text {fmt(x['cot_same_as_gt_nav'], 1, True)} %",
              "- ADE difference by motion: " + ", ".join(f"{k} {v['d_ade_gt']:+.2f} (n={v['n']})" for k, v in x["by_motion"].items())]
    L += ["", "## Reasoning", "", "| | CoT match at same seed | Length difference (tokens) | NOMINAL share difference | Median divergence position |", "|---|---|---|---|---|",
          f"| BF16 vs BF16, different seed | {fmt(ov['bf16_seed_cot_same'], 1, True)} % | (BF16 length {fmt(ov['base_ntok'], 1)}) | (BF16 {fmt(ov['base_nominal'], 1, True)} %) | |"]
    for v in others:
        L.append(f"| {v} vs BF16 | {fmt(ov[v]['same_seed_cot'], 1, True)} % | {fmt(ov[v]['d_ntok'], 2)} | {fmt(ov[v]['d_nominal'], 1, True)} pt | {ov[v]['div_pos_median']} |")
    L += ["", "## Teacher forcing (CoT fixed to BF16 seed 43)", "", "| Variant | KL (T=1) | KL ÷ BF16 rounding jitter (eager) | Top-1 match | Swaps | Trajectory difference ÷ diffusion noise spread |", "|---|---|---|---|---|---|"]
    for v in tfv:
        t = ov[f"tf_{v}"]
        L.append(f"| {v} | {fmt(t['kl_t1'], 6)} | {fmt(t.get('kl_ratio_vs_eager'), 2)} | {fmt(t['top1'], 2, True)} % | {t['top1_miss_total']}/{t['n_tok_total']} | {fmt(t['traj_ratio_vs_noise'], 2, True)} % |")
    L += ["", "## Per scene category (FP8 same-seed trajectory difference ÷ BF16 seed spread)", ""]
    for col, d in cats.items():
        L += [f"### {col}", "", "| Category | Count (logs) | BF16 seed spread m | " + " | ".join(f"{v} ratio / CoT match / ΔADE / KL" for v in others) + " |",
              "|---|---|---|" + "---|" * len(others)]
        for k, e in d.items():
            cells = []
            for v in others:
                x = e[v]
                ci = f" [{x['ratio_ci'][1]*100:.0f}, {x['ratio_ci'][2]*100:.0f}]" if x["ratio_ci"] else ""
                cells.append(f"{x['ratio']*100:.1f}%{ci} / {x['same_cot']*100:.0f}% / {x['d_ade_gt']:+.3f} / {x['tf_kl']:.1e}")
            L.append(f"| {k} | {e['n']} ({e['n_clips']}) | {e['bf16_seed_traj']:.3f} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Speed (single-input measurement, including ViT, runs that built a new graph excluded)", "",
          "| Variant | Runs | ms per token (slope) | Time median | 90% | Max | Under 1 s | Length median / 90% / max | Length that fits in 1 s (regression) |", "|---|---|---|---|---|---|---|---|---|"]
    for v, e in spd.items():
        L.append(f"| {v} | {e['n_calls']} | {e['ms_per_token_slope']:.2f} | {e['e2e_median']:.0f} | {e['e2e_p90']:.0f} | {e['e2e_max']:.0f} | "
                 f"{e['under_1s']*100:.1f}% | {e['n_tok_median']:.0f} / {e['n_tok_p90']:.0f} / {e['n_tok_max']} | {e.get('tokens_within_1s_fit', float('nan')):.0f} |")
    L += ["", f"BF16 CoT length distribution (tokens, bins {hist[1].tolist()}): {hist[0].tolist()}", "",
          f"Runs excluded for building a new graph: {res['excluded_new_graph_calls']}, max temperature {res['temp_max']}°C"]
    open(a.out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

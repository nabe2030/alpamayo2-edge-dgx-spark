#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# summarize_wide.py <wide.jsonl> <scenes.csv> <out_prefix> [--ref-1clip <summary_fp8.json>]
#   wide_eval.py の記録を集計する。出力: <out_prefix>.json(数値)と <out_prefix>.md(表)。
#
# 物差し: BF16(base)同士で seed だけを変えたときの差(seed の組 i<j)。FP8 の変種は、同じ場面・同じ seed の base と比べる。
# 信頼区間: 同じクリップの t0 どうしは似ているので、クリップ単位のブートストラップ(2,000 回、seed 0)で 95% 区間を出す。
# AV2: ここでの「クリップ」は AV2 のログ(clip_id = log_id)。同じログの 2 場面はまとめて抽出する。
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
    """記録された未来の軌跡(t0 の自車座標、x 前・y 左)から、行動の種類を大まかに決める。"""
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
    """クリップ単位のブートストラップで stat(df[col]) の 95% 区間。"""
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
    """mean(num) / mean(den) の 95% 区間(クリップ単位)。"""
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
    scenes = pd.read_csv(a.scenes)  # AV2: 分類は記録(sample)の側に入っているので、ここでは使わない
    variants = meta["variants"]
    others = variants[1:]
    seeds = sorted(samples[0]["sampled"]["base"].keys(), key=int)
    tfv = meta["tf_variants"]

    rows, speed = [], []
    for r in samples:
        base = r["sampled"]["base"]
        gt = r["gt_xyz"]
        row = {"sample_id": r["sample_id"], "clip_id": r["clip_id"], "nav_text": r["nav_text"],
               # AV2: 走り方(場面選びでの分類)、都市、split。nav_group は gt のナビ文の操作
               "motion": r.get("motion"), "city": r.get("city"), "split": r.get("split"),
               "nav_group": (r["nav_text"].split(" in ")[0] if r.get("nav_text") else None),
               "behavior": behavior(gt), "temp_max": r.get("temp_max")}
        # BF16 同士(seed 違い)の幅
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
        # AV2: 参考。BF16 で、ナビ文をいつも "Continue straight" にしたとき
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

    # ---- 全体
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

    # ---- 場面の種類ごと
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

    # ---- 差が大きい場面(FP8 の同 seed の軌跡差 ÷ BF16 の seed の幅、と KL の上位)
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

    # ---- 速さ(1 回だけの測り方。新しいグラフを作った回は除く)
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

    # ---- 表(md)
    L = [f"# 広いデータでの比較の集計(AV2、{res['n_samples']} 場面 / {res['n_clips']} ログ、seed {','.join(seeds)})", ""]
    L += ["## 軌跡(m、平均 [ログ単位ブートストラップ 95% 区間])", "",
          "| 比べるもの | 値 |", "|---|---|",
          f"| BF16 同士、seed だけ違う(seed の組の平均) | {fmt(ov['bf16_seed_traj_mean'])} |",
          f"| BF16、同じ CoT・拡散の乱数だけ違う(teacher forcing) | {fmt(ov['tf_noise_alt_mean'])} |"]
    for v in others:
        L.append(f"| {v} 対 BF16、同じ seed | {fmt(ov[v]['same_seed_traj_mean'])}(BF16 の seed の幅の {fmt(ov[v]['ratio_traj_vs_bf16_seed'], 1, True)} %) |")
    L += ["", "| 記録された軌跡との誤差 | ADE(seed 平均) | 差(変種 − BF16) | minADE(5 seed) の差 |", "|---|---|---|---|",
          f"| BF16 | {fmt(ov['base_ade_gt'])} | — | — |"]
    for v in others:
        L.append(f"| {v} | {fmt(ov[v]['ade_gt'])} | {fmt(ov[v]['d_ade_gt'])} | {fmt(ov[v]['d_minade_gt'])} |")
    if "straight" in ov:
        x = ov["straight"]
        L += ["", "### 参考: ナビ文をいつも \"Continue straight\" にしたとき(BF16)", "",
              f"- ADE {fmt(x['ade_gt'])}、gt のナビ文との差 {fmt(x['d_ade_gt'])}、minADE {fmt(x['minade_gt'])}、CoT が gt のナビ文と同じ {fmt(x['cot_same_as_gt_nav'], 1, True)} %",
              "- 走り方ごとの ADE の差: " + "、".join(f"{k} {v['d_ade_gt']:+.2f}(n={v['n']})" for k, v in x["by_motion"].items())]
    L += ["", "## 判断理由", "", "| | 同じ seed で CoT 一致 | 長さの差(トークン) | NOMINAL の割合の差 | 分かれる位置の中央値 |", "|---|---|---|---|---|",
          f"| BF16 同士、seed 違い | {fmt(ov['bf16_seed_cot_same'], 1, True)} % | (BF16 の長さ {fmt(ov['base_ntok'], 1)}) | (BF16 {fmt(ov['base_nominal'], 1, True)} %) | |"]
    for v in others:
        L.append(f"| {v} 対 BF16 | {fmt(ov[v]['same_seed_cot'], 1, True)} % | {fmt(ov[v]['d_ntok'], 2)} | {fmt(ov[v]['d_nominal'], 1, True)} pt | {ov[v]['div_pos_median']} |")
    L += ["", "## teacher forcing(BF16 の seed 43 の CoT を固定)", "", "| 変種 | KL(T=1) | KL ÷ BF16 の丸めの揺れ(eager) | 上位 1 位の一致 | 入れ替わり | 軌跡差 ÷ 拡散の乱数の幅 |", "|---|---|---|---|---|---|"]
    for v in tfv:
        t = ov[f"tf_{v}"]
        L.append(f"| {v} | {fmt(t['kl_t1'], 6)} | {fmt(t.get('kl_ratio_vs_eager'), 2)} | {fmt(t['top1'], 2, True)} % | {t['top1_miss_total']}/{t['n_tok_total']} | {fmt(t['traj_ratio_vs_noise'], 2, True)} % |")
    L += ["", "## 場面の種類ごと(FP8 の同 seed の軌跡差 ÷ BF16 の seed の幅)", ""]
    for col, d in cats.items():
        L += [f"### {col}", "", "| 種類 | 件数(ログ) | BF16 seed の幅 m | " + " | ".join(f"{v} 比 / CoT 一致 / ΔADE / KL" for v in others) + " |",
              "|---|---|---|" + "---|" * len(others)]
        for k, e in d.items():
            cells = []
            for v in others:
                x = e[v]
                ci = f" [{x['ratio_ci'][1]*100:.0f}, {x['ratio_ci'][2]*100:.0f}]" if x["ratio_ci"] else ""
                cells.append(f"{x['ratio']*100:.1f}%{ci} / {x['same_cot']*100:.0f}% / {x['d_ade_gt']:+.3f} / {x['tf_kl']:.1e}")
            L.append(f"| {k} | {e['n']}({e['n_clips']}) | {e['bf16_seed_traj']:.3f} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## 速さ(1 回だけの測り方、ViT 込み、新しいグラフを作った回は除く)", "",
          "| 変種 | 回数 | ms/トークン(傾き) | 時間 中央値 | 90% | 最大 | 1 秒未満 | 長さ 中央値 / 90% / 最大 | 1 秒に収まる長さ(回帰) |", "|---|---|---|---|---|---|---|---|---|"]
    for v, e in spd.items():
        L.append(f"| {v} | {e['n_calls']} | {e['ms_per_token_slope']:.2f} | {e['e2e_median']:.0f} | {e['e2e_p90']:.0f} | {e['e2e_max']:.0f} | "
                 f"{e['under_1s']*100:.1f}% | {e['n_tok_median']:.0f} / {e['n_tok_p90']:.0f} / {e['n_tok_max']} | {e.get('tokens_within_1s_fit', float('nan')):.0f} |")
    L += ["", f"BF16 の CoT の長さの分布(トークン、区間 {hist[1].tolist()}): {hist[0].tolist()}", "",
          f"新しいグラフを作って除いた回: {res['excluded_new_graph_calls']}、最高温度 {res['temp_max']}℃"]
    open(a.out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

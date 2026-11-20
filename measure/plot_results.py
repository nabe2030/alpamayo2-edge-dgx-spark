#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# plot_results.py <speed_summary.json> <wide_summary.json> <out_dir>
#  の図(図用の venv で動かす: matplotlib 3.9.2 + japanize-matplotlib)。
#   fig_speed_single.png : 1 回だけの時間を、段階ごとに積んだ横棒(公式の加速 → 4 つの手 → FP8)。1 秒の線つき
#   fig_speed_stream.png : 連続 10Hz / 1Hz の時間の分布(中央値・p90・最大)。1 秒の線つき
#   fig_fp8_vs_bf16.png  : 同じ seed での軌跡の差 ÷ BF16 の seed の幅(全体と走り方ごと、95% 区間)
#   fig_ms_per_token.png : 1 トークンあたりの時間と、1 秒に収まるトークン数
# 色は dataviz の参照パレットの順番どおり(段階: 5 色、変種: 先頭 3 色 = 全組み合わせで色覚の検査を通る範囲)。
import json
import sys
from pathlib import Path

import japanize_matplotlib  # noqa: F401  日本語のフォント
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
VAR = {"base": ("BF16", PAL[0]), "fp8body": ("本体だけ FP8", PAL[1]), "fp8all": ("本体+出力層 FP8", PAL[2])}
CREDIT = "AV2 (CC BY-NC-SA 4.0, © 2021 Argo AI, LLC) で測定"


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, path):
    # 帰属表示は図の下の余白(tight_layout の rect で空けた所)に置き、軸の説明と重ねない
    fig.text(0.99, 0.005, CREDIT, ha="right", va="bottom", fontsize=7, color=INK2)
    fig.savefig(path, dpi=150, facecolor=SURF)
    plt.close(fig)
    print("保存", path)


def speed_single(S, out):
    order = [("A_off_s43", "公式・加速なし"), ("A_compile_s43", "公式 compile"), ("A_tuned_s43", "公式 tuned"),
             ("B_p_s43", "+ #1 処理器を 1 回だけ"), ("B_pg_s43", "+ #9 GPU で縮小"), ("B_pgc_s43", "+ #2 ViT の使い回し"),
             ("B_pgcv_s43", "+ #3 ViT の compile"), ("C_body_s43", "+ 本体だけ FP8"), ("C_all_s43", "+ 本体+出力層 FP8")]
    order = [(k, n) for k, n in order if k in S["single"]]
    stages = [("preprocess", "前処理"), ("vision", "ViT"), ("prefill_llm", "プリフィル"), ("decode", "判断理由のデコード"),
              ("expert", "軌跡(拡散 10 ステップ)")]
    fig, ax = plt.subplots(figsize=(9, 0.48 * len(order) + 1.6))
    style(ax)
    for i, (k, name) in enumerate(order):
        v = S["single"][k]
        left = 0.0
        for j, (s, _) in enumerate(stages):
            w = v["stages"].get(s) or 0.0
            ax.barh(i, w, left=left, height=0.62, color=PAL[j], edgecolor=SURF, linewidth=1.5)
            left += w
        ax.text(v["e2e_median"] + 15, i, f"{v['e2e_median']:.0f} ms", va="center", fontsize=8.5, color=INK,
                bbox=dict(facecolor=SURF, edgecolor="none", pad=1.5), zorder=5)
    ax.axvline(1000, color=INK2, linewidth=1, linestyle="--", zorder=1)
    ax.text(1000, -0.75, "1 秒", fontsize=8, color=INK2, ha="center", va="bottom")
    ax.set_yticks(range(len(order)), [n for _, n in order], fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("1 回の推論にかかる時間(ms、20 回の中央値。棒は段階ごとの中央値を積んだもの、右の数字は全体の中央値)",
                  fontsize=8.5, color=INK2)
    tok = S["single"][order[0][0]]["tokens"]
    ax.set_title(f"1 回だけの推論の時間(AV2 の 1 場面、判断理由 {tok:.0f} トークン、seed 43)", fontsize=11, color=INK, loc="left",
                 pad=16)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PAL[j]) for j in range(len(stages))]
    fig.legend(handles, [n for _, n in stages], loc="lower center", bbox_to_anchor=(0.5, 0.035),
               ncol=5, fontsize=8.5, frameon=False)
    xmax = max(S["single"][k]["e2e_median"] for k, _ in order)
    ax.set_xlim(0, max(1100, xmax * 1.12))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save(fig, out)


def speed_stream(S, out):
    order = [("S10_tuned", "10Hz 公式 tuned"), ("S10_p", "10Hz + #1"), ("S10_pg", "10Hz + #9"), ("S10_pgc", "10Hz + #2"),
             ("S10_pgv", "10Hz + #3(#2 なし)"), ("S10_pgcv", "10Hz 4 つの手"), ("S10_body", "10Hz + 本体 FP8"),
             ("S10_all", "10Hz + 本体+出力層 FP8"),
             ("S1_pgv", "1Hz #1#9#3"), ("S1_pgcv", "1Hz 4 つの手"), ("S1_body", "1Hz + 本体 FP8"), ("S1_all", "1Hz + 本体+出力層 FP8")]
    order = [(k, n) for k, n in order if k in S["stream"]]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(order) + 1.6))
    style(ax)
    for i, (k, name) in enumerate(order):
        v = S["stream"][k]
        ax.plot([v["p10"], v["max"]], [i, i], color=GRID, linewidth=2, solid_capstyle="round", zorder=1)
        ax.plot([v["e2e_median"], v["p90"]], [i, i], color=PAL[0], linewidth=4, solid_capstyle="round", zorder=2)
        ax.scatter([v["e2e_median"]], [i], s=46, color=PAL[0], edgecolor=SURF, linewidth=1.5, zorder=3)
        ax.scatter([v["max"]], [i], s=40, marker="D", color=PAL[1], edgecolor=SURF, linewidth=1.5, zorder=3)
        ax.text(v["max"] + 35, i, f"中央値 {v['e2e_median']:.0f} / 最大 {v['max']:.0f}  1 秒超え {100 * v['frac_over_1s']:.1f}%",
                va="center", fontsize=8, color=INK, bbox=dict(facecolor=SURF, edgecolor="none", pad=1.5), zorder=5)
    ax.axvline(1000, color=INK2, linewidth=1, linestyle="--", zorder=0)
    ax.set_yticks(range(len(order)), [n for _, n in order], fontsize=9, color=INK)
    ax.invert_yaxis()
    xmax = max(S["stream"][k]["max"] for k, _ in order)
    ax.set_xlim(0, max(1100, xmax) * 1.45)
    ax.set_xlabel("1 歩の時間(ms)。灰の線 = p10〜最大、青の太線 = 中央値〜p90、● = 中央値、◆ = 最大、点線 = 1 秒",
                  fontsize=8.5, color=INK2)
    ax.set_title("連続入力の 1 歩ごとの時間(同じログの t0 を 0.1 s / 1 s ずつ進める)", fontsize=11, color=INK, loc="left")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save(fig, out)


def fp8_ratio(W, out):
    ov, cats = W["overall"], W["by_category"]
    groups = [("全体", None)] + [(f"{k}", ("motion", k)) for k in ("left", "right", "straight", "stop", "other")
                                if k in cats.get("motion", {})]
    names = {"left": "左折", "right": "右折", "straight": "直進", "stop": "停止", "other": "その他"}
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    w = 0.34
    for j, v in enumerate(("fp8body", "fp8all")):
        xs, ys, lo, hi = [], [], [], []
        for i, (g, key) in enumerate(groups):
            if key is None:
                r = ov[v]["ratio_traj_vs_bf16_seed"]
                m, a, b = r[0], r[1], r[2]
            else:
                e = cats[key[0]][key[1]][v]
                m = e["ratio"]
                a, b = (e["ratio_ci"][1], e["ratio_ci"][2]) if e.get("ratio_ci") else (m, m)
            xs.append(i + (j - 0.5) * w)
            ys.append(m * 100)
            lo.append((m - a) * 100)
            hi.append((b - m) * 100)
        ax.bar(xs, ys, width=w - 0.04, color=VAR[v][1], edgecolor=SURF, linewidth=1.5, label=VAR[v][0])
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
    labels = []
    for g, key in groups:
        if key is None:
            labels.append(f"全体\n(n={W['n_samples']})")
        else:
            labels.append(f"{names.get(key[1], key[1])}\n(n={cats[key[0]][key[1]]['n']})")
    ax.set_xticks(range(len(groups)), labels, fontsize=9, color=INK)
    ax.set_ylabel("同じ seed の軌跡の差 ÷ BF16 の seed 違いの幅(%)", fontsize=9, color=INK2)
    ax.set_ylim(bottom=0)
    ax.set_title("FP8 にしたときの軌跡の変わり方(BF16 どうしで seed を変えたときの幅を 100% とする、95% 区間つき)",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save(fig, out)


def ms_per_token(S, W, out):
    rows = []
    for v, k in (("base", "S10_pgcv"), ("fp8body", "S10_body"), ("fp8all", "S10_all")):
        if k in S["fit"]:
            rows.append((v, S["fit"][k]["decode_slope"], S["fit"][k]["max_tokens_1s"]))
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, idx, title, unit in ((axes[0], 1, "1 トークンあたりの時間", "ms/トークン"),
                                 (axes[1], 2, "1 秒に収まる判断理由の長さ", "トークン")):
        style(ax)
        for i, r in enumerate(rows):
            ax.barh(i, r[idx], height=0.6, color=VAR[r[0]][1], edgecolor=SURF, linewidth=1.5)
            ax.text(r[idx] * 1.02, i, f"{r[idx]:.2f}" if idx == 1 else f"{r[idx]:.0f}", va="center", fontsize=9, color=INK)
        ax.set_yticks(range(len(rows)), [VAR[r[0]][0] for r in rows], fontsize=9, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, max(r[idx] for r in rows) * 1.25)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel(unit + "(連続 10Hz の歩をトークン数に直線で当てはめた値)", fontsize=8, color=INK2)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, out)


def main(speed_json, wide_json, out_dir):
    S = json.loads(Path(speed_json).read_text())
    W = json.loads(Path(wide_json).read_text()) if Path(wide_json).exists() else None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    speed_single(S, out / "fig_speed_single.png")
    speed_stream(S, out / "fig_speed_stream.png")
    ms_per_token(S, W, out / "fig_ms_per_token.png")
    if W:
        fp8_ratio(W, out / "fig_fp8_vs_bf16.png")


if __name__ == "__main__":
    main(*sys.argv[1:])

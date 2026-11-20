#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# plot_results.py <speed_summary.json> <wide_summary.json> <out_dir>
#  figures (run in the plotting venv: matplotlib 3.9.2 + japanize-matplotlib).
#   fig_speed_single.png : single-input time as horizontal bars stacked by stage (official acceleration → the four optimizations → FP8). With a 1 s line
#   fig_speed_stream.png : distribution of continuous 10Hz / 1Hz times (median, p90, max). With a 1 s line
#   fig_fp8_vs_bf16.png  : same-seed trajectory difference ÷ BF16 seed spread (overall and per motion, 95% intervals)
#   fig_ms_per_token.png : time per token, and number of tokens that fit in 1 s
# Colors follow the order of the dataviz reference palette (stages: 5 colors; variants: first 3 colors = the range that passes the color-vision check for all combinations).
import json
import sys
from pathlib import Path

import japanize_matplotlib  # noqa: F401  Japanese font
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
VAR = {"base": ("BF16", PAL[0]), "fp8body": ("FP8 body only", PAL[1]), "fp8all": ("FP8 body + output layer", PAL[2])}
CREDIT = "Measured on AV2 (CC BY-NC-SA 4.0, © 2021 Argo AI, LLC)"


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
    # Put the attribution in the bottom margin (the space left by tight_layout's rect) so it does not overlap the axis labels
    fig.text(0.99, 0.005, CREDIT, ha="right", va="bottom", fontsize=7, color=INK2)
    fig.savefig(path, dpi=150, facecolor=SURF)
    plt.close(fig)
    print("saved", path)


def speed_single(S, out):
    order = [("A_off_s43", "Official, no acceleration"), ("A_compile_s43", "Official compile"), ("A_tuned_s43", "Official tuned"),
             ("B_p_s43", "+ #1 load processor once"), ("B_pg_s43", "+ #9 resize on GPU"), ("B_pgc_s43", "+ #2 reuse ViT results"),
             ("B_pgcv_s43", "+ #3 compile ViT too"), ("C_body_s43", "+ FP8 body only"), ("C_all_s43", "+ FP8 body + output layer")]
    order = [(k, n) for k, n in order if k in S["single"]]
    stages = [("preprocess", "Preprocessing"), ("vision", "ViT"), ("prefill_llm", "Prefill"), ("decode", "Reasoning decode"),
              ("expert", "Trajectory (10 diffusion steps)")]
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
    ax.text(1000, -0.75, "1 s", fontsize=8, color=INK2, ha="center", va="bottom")
    ax.set_yticks(range(len(order)), [n for _, n in order], fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Time per inference (ms, median of 20 runs; bars stack per-stage medians, number on the right is the overall median)",
                  fontsize=8.5, color=INK2)
    tok = S["single"][order[0][0]]["tokens"]
    ax.set_title(f"Single-input inference time (1 AV2 scene, {tok:.0f} reasoning tokens, seed 43)", fontsize=11, color=INK, loc="left",
                 pad=16)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PAL[j]) for j in range(len(stages))]
    fig.legend(handles, [n for _, n in stages], loc="lower center", bbox_to_anchor=(0.5, 0.035),
               ncol=5, fontsize=8.5, frameon=False)
    xmax = max(S["single"][k]["e2e_median"] for k, _ in order)
    ax.set_xlim(0, max(1100, xmax * 1.12))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save(fig, out)


def speed_stream(S, out):
    order = [("S10_tuned", "10Hz official tuned"), ("S10_p", "10Hz + #1"), ("S10_pg", "10Hz + #9"), ("S10_pgc", "10Hz + #2"),
             ("S10_pgv", "10Hz + #3 (no #2)"), ("S10_pgcv", "10Hz four optimizations"), ("S10_body", "10Hz + FP8 body only"),
             ("S10_all", "10Hz + FP8 body + output layer"),
             ("S1_pgv", "1Hz #1#9#3"), ("S1_pgcv", "1Hz four optimizations"), ("S1_body", "1Hz + FP8 body only"), ("S1_all", "1Hz + FP8 body + output layer")]
    order = [(k, n) for k, n in order if k in S["stream"]]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(order) + 1.6))
    style(ax)
    for i, (k, name) in enumerate(order):
        v = S["stream"][k]
        ax.plot([v["p10"], v["max"]], [i, i], color=GRID, linewidth=2, solid_capstyle="round", zorder=1)
        ax.plot([v["e2e_median"], v["p90"]], [i, i], color=PAL[0], linewidth=4, solid_capstyle="round", zorder=2)
        ax.scatter([v["e2e_median"]], [i], s=46, color=PAL[0], edgecolor=SURF, linewidth=1.5, zorder=3)
        ax.scatter([v["max"]], [i], s=40, marker="D", color=PAL[1], edgecolor=SURF, linewidth=1.5, zorder=3)
        ax.text(v["max"] + 35, i, f"median {v['e2e_median']:.0f} / max {v['max']:.0f}  over 1 s {100 * v['frac_over_1s']:.1f}%",
                va="center", fontsize=8, color=INK, bbox=dict(facecolor=SURF, edgecolor="none", pad=1.5), zorder=5)
    ax.axvline(1000, color=INK2, linewidth=1, linestyle="--", zorder=0)
    ax.set_yticks(range(len(order)), [n for _, n in order], fontsize=9, color=INK)
    ax.invert_yaxis()
    xmax = max(S["stream"][k]["max"] for k, _ in order)
    ax.set_xlim(0, max(1100, xmax) * 1.45)
    ax.set_xlabel("Time per step (ms). Gray line = p10–max, thick blue = median–p90, ● = median, ◆ = max, dashed = 1 s",
                  fontsize=8.5, color=INK2)
    ax.set_title("Per-step time, continuous input (t0 of one log advanced by 0.1 s / 1 s)", fontsize=11, color=INK, loc="left")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save(fig, out)


def fp8_ratio(W, out):
    ov, cats = W["overall"], W["by_category"]
    groups = [("Overall", None)] + [(f"{k}", ("motion", k)) for k in ("left", "right", "straight", "stop", "other")
                                if k in cats.get("motion", {})]
    names = {"left": "Left turn", "right": "Right turn", "straight": "Straight", "stop": "Stopped", "other": "Other"}
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
            labels.append(f"Overall\n(n={W['n_samples']})")
        else:
            labels.append(f"{names.get(key[1], key[1])}\n(n={cats[key[0]][key[1]]['n']})")
    ax.set_xticks(range(len(groups)), labels, fontsize=9, color=INK)
    ax.set_ylabel("Same-seed trajectory diff ÷ BF16 seed spread (%)", fontsize=9, color=INK2)
    ax.set_ylim(bottom=0)
    ax.set_title("Trajectory change with FP8 (BF16 spread across seeds = 100%, with 95% intervals)",
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
    for ax, idx, title, unit in ((axes[0], 1, "Time per token", "ms per token"),
                                 (axes[1], 2, "Reasoning length that fits in 1 s", "tokens")):
        style(ax)
        for i, r in enumerate(rows):
            ax.barh(i, r[idx], height=0.6, color=VAR[r[0]][1], edgecolor=SURF, linewidth=1.5)
            ax.text(r[idx] * 1.02, i, f"{r[idx]:.2f}" if idx == 1 else f"{r[idx]:.0f}", va="center", fontsize=9, color=INK)
        ax.set_yticks(range(len(rows)), [VAR[r[0]][0] for r in rows], fontsize=9, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, max(r[idx] for r in rows) * 1.25)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel(unit + " (linear fit to continuous 10Hz steps)", fontsize=8, color=INK2)
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

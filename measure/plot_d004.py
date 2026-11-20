#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# plot_d004.py <multi.json> <multi_body.json> <e2e12.json> <e2e12_body.json> <out_dir>
# Figures (figure venv). Redraws the counterparts of fig_multi_stream and fig_e2e of plot_d003.py with three configurations,
# adding FP8 body only. Style and colors follow plot_results.py.
#   fig_multi_stream_3cfg.png : continuous time per log (inference only; dot = median, line = max). 10 Hz / 1 Hz × 3 configurations, with a 1 s line
#   fig_e2e12_3cfg.png        : time per step including re-rendering, 1 Hz, 12 logs (re-rendering + inference medians stacked, max as a vertical line)
import json
import sys
from pathlib import Path

import japanize_matplotlib  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_results import INK, INK2, PAL, SURF, save, style  # noqa: E402

COND = {"pgcv": ("Four optimizations (BF16)", PAL[0]), "body": ("+ FP8 body only", PAL[1]),
        "all": ("+ FP8 body + output layer", PAL[2])}


def multi(C, out):
    logs = list(dict.fromkeys(l for c in C.values() for l in c["per_log"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.45 * len(logs) + 2.6), sharey=True)
    for ax, hz in zip(axes, ("10", "1")):
        style(ax)
        for j, (k, (name, col)) in enumerate(COND.items()):
            c = C.get(f"M{hz}_{k}")
            if not c:
                continue
            for i, l in enumerate(logs):
                p = c["per_log"].get(l)
                if not p:
                    continue
                y = i + (j - 1) * 0.26
                ax.plot([p["median"], p["max"]], [y, y], color=col, linewidth=2, solid_capstyle="round")
                ax.plot(p["median"], y, "o", color=col, markersize=6, markeredgecolor=SURF, markeredgewidth=1.2,
                        label=name if i == 0 else None)
            ax.text(0.99, -0.11 - 0.055 * j, f"{name}: {c['n']} steps, median {c['median']:.0f} / max {c['max']:.0f} ms, "
                    f"over 1 s {100 * c['over_1s']:.1f}%", transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
        ax.axvline(1000, color=INK2, linestyle="--", linewidth=1)
        ax.text(1000, len(logs) - 0.3, " 1 s", color=INK2, fontsize=8, va="bottom")
        ax.set_xlim(0, 1150)
        ax.set_title(f"Continuous every {hz} Hz (inference only, re-rendering excluded)", fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("Time per step, ms (dot = median, right end of line = max)", fontsize=9, color=INK2)
    axes[0].set_yticks(range(len(logs)), [l[:8] for l in logs])
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower left", fontsize=8.5, frameon=True, facecolor=SURF, edgecolor=SURF)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    save(fig, out / "fig_multi_stream_3cfg.png")


def e2e(C, out):
    rows = [(k, C[f"F1_{k}"]) for k in COND if f"F1_{k}" in C]
    fig, ax = plt.subplots(figsize=(10, 0.75 * len(rows) + 1.9))
    style(ax)
    for i, (k, c) in enumerate(rows):
        r, e = c["render_ms"]["median"], c["e2e_ms"]["median"]
        ax.barh(i, r, height=0.6, color=PAL[3], edgecolor=SURF, linewidth=1.5, label="Re-rendering (from reading the JPEGs)" if i == 0 else None)
        ax.barh(i, e, left=r, height=0.6, color=COND[k][1], edgecolor=SURF, linewidth=1.5)
        ax.text(r + e / 2, i, "Inference", va="center", ha="center", fontsize=8, color=SURF)
        ax.plot([c["max"], c["max"]], [i - 0.3, i + 0.3], color=INK, linewidth=2)
        ax.text(max(c["max"], 1000) + 14, i, f"median {c['median']:.0f} / max {c['max']:.0f} ms\nover 1 s {100 * c['over_1s']:.1f}%"
                f" ({c['n']} steps)", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)), [COND[k][0] for k, _ in rows], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(1000, color=INK2, linestyle="--", linewidth=1)
    ax.text(1000, -0.42, " 1 s", color=INK2, fontsize=8, va="top")
    ax.set_xlim(0, 1500)
    ax.set_xlabel("Time per step, ms (bar = breakdown of medians, vertical line = max of total)", fontsize=9, color=INK2)
    ax.set_title("Continuous including JPEG reading and re-rendering (1 Hz, 12 logs, run back to back on the same GPU)", fontsize=10.5, color=INK, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out / "fig_e2e12_3cfg.png")


def main():
    m, mb, e, eb, out = sys.argv[1:6]
    out = Path(out)
    M = {**json.loads(Path(m).read_text())["conditions"], **json.loads(Path(mb).read_text())["conditions"]}
    multi(M, out)
    E = {**json.loads(Path(e).read_text())["conditions"], **json.loads(Path(eb).read_text())["conditions"]}
    e2e(E, out)


if __name__ == "__main__":
    main()

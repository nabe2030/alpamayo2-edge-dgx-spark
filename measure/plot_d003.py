#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# plot_d003.py <multi.json> <e2e.json|-> <stopgo.json|-> <out_dir>
#  figures (plotting venv: matplotlib + japanize-matplotlib). Layout and colors match plot_results.py.
#   fig_multi_stream.png : continuous time per log (dot = median, line = max). 10Hz / 1Hz × the four optimizations / + FP8. With a 1 s line
#   fig_e2e.png          : per-step time including re-rendering (medians of re-rendering + inference stacked, max as a line). With a 1 s line
#   fig_stopgo.png       : share of "go" in stopped scenes (per variant, 95% intervals), and the difference in first-token "go" probability
import json
import sys
from pathlib import Path

import japanize_matplotlib  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_results import GRID, INK, INK2, PAL, SURF, VAR, save, style  # noqa: E402

COND = {"pgcv": ("Four optimizations (BF16)", PAL[0]), "all": ("+ FP8 body + output layer", PAL[2])}


def multi(M, out):
    C = M["conditions"]
    logs = list(dict.fromkeys(l for c in C.values() for l in c["per_log"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.36 * len(logs) + 2.0), sharey=True)
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
                y = i + (j - 0.5) * 0.3
                ax.plot([p["median"], p["max"]], [y, y], color=col, linewidth=2, solid_capstyle="round")
                ax.plot(p["median"], y, "o", color=col, markersize=7, markeredgecolor=SURF, markeredgewidth=1.5,
                        label=name if i == 0 else None)
            ax.text(0.99, -0.13 - 0.06 * j, f"{name}: {c['n']} steps, median {c['median']:.0f} / max {c['max']:.0f} ms, "
                    f"over 1 s {100 * c['over_1s']:.1f}%", transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
        ax.axvline(1000, color=INK2, linestyle="--", linewidth=1)
        ax.text(1000, len(logs) - 0.3, " 1 s", color=INK2, fontsize=8, va="bottom")
        ax.set_xlim(0, 1150)
        ax.set_title(f"Continuous {hz}Hz (inference only, no re-rendering)", fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("Time per step, ms (dot = median, right end of line = max)", fontsize=9, color=INK2)
    axes[0].set_yticks(range(len(logs)), [l[:8] for l in logs])
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower left", fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, out / "fig_multi_stream.png")


def e2e(E, out):
    C = E["conditions"]
    order = {"E1_all": 0, "E1_pgcv": 1, "E10_all": 2, "E10_pgcv": 3}
    rows = sorted([(k, c) for k, c in C.items() if c.get("metric") == "total_ms"],
                  key=lambda kc: (order.get("_".join(kc[0].split("_")[:2]), 9), kc[0].endswith("_cpu")))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(10, 0.62 * len(rows) + 1.9))
    style(ax)
    for i, (k, c) in enumerate(rows):
        r, e = c["render_ms"]["median"], c["e2e_ms"]["median"]
        ax.barh(i, r, height=0.6, color=PAL[3], edgecolor=SURF, linewidth=1.5, label="Re-rendering (from JPEG decode)" if i == 0 else None)
        ax.barh(i, e, left=r, height=0.6, color=PAL[0], edgecolor=SURF, linewidth=1.5, label="Inference" if i == 0 else None)
        ax.plot([c["max"], c["max"]], [i - 0.3, i + 0.3], color=INK, linewidth=2)
        ax.text(c["max"] + 12, i, f"median {c['median']:.0f} / max {c['max']:.0f} ms, over 1 s {100 * c['over_1s']:.1f}%",
                va="center", fontsize=8, color=INK)
    names = {"E1": "1Hz", "E10": "10Hz"}

    def label(k):
        hz, cond = names.get(k.split("_")[0], k), COND.get(k.split("_")[1], (k,))[0]
        how = "per-image .cpu() (v1)" if k.endswith("_cpu") else "pinned batch copy"
        return f"{hz} {cond}\nre-rendering: {how}"
    ax.set_yticks(range(len(rows)), [label(k) for k, _ in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.axvline(1000, color=INK2, linestyle="--", linewidth=1)
    ax.set_xlim(0, max(1500, max(c["max"] for _, c in rows) + 420))
    ax.set_xlabel("Time per step, ms (bars = median breakdown, vertical line = max of total)", fontsize=9, color=INK2)
    ax.set_title("Continuous time incl. re-rendering (back to back on the same GPU)", fontsize=10.5, color=INK, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out / "fig_e2e.png")


def stopgo(G, out):
    S = G["sets"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    ax = axes[0]
    style(ax)
    sets = [("A", "A: stopped in the recorded drive\n(go = trajectory moves 2 m+ in 3 s)", "traj_go"),
            ("B", "B: reasoning says \"stop\"\n(share of reasoning = go)", "cot_go")]
    for i, (s, label, key) in enumerate(sets):
        if s not in S:
            continue
        for j, v in enumerate(("base", "fp8body", "fp8all")):
            m, (lo, hi) = S[s]["rates"][v][key]
            y = i + (j - 1) * 0.22
            ax.plot([100 * lo, 100 * hi], [y, y], color=VAR[v][1], linewidth=2)
            ax.plot(100 * m, y, "o", color=VAR[v][1], markersize=7, markeredgecolor=SURF, markeredgewidth=1.5,
                    label=VAR[v][0] if i == 0 else None)
            ax.text(100 * hi + 1, y, f"{100 * m:.1f}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(sets)), [l for _, l, _ in sets], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(10.0, ax.get_xlim()[1] + 6))
    ax.set_xlabel("Share of \"go\", % (dot = overall, line = scene-resampled 95% interval)", fontsize=9, color=INK2)
    ax.set_title("Share of \"go\" in stopped scenes", fontsize=10.5, color=INK, loc="left")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax = axes[1]
    style(ax)
    F = G.get("first_token", {})
    k = 0
    labels = []
    for s in ("A", "B"):
        for v in ("fp8body", "fp8all"):
            d = F.get(s, {}).get(v, {}).get("diff_go")
            if not d:
                continue
            ax.plot([100 * d["ci"][0], 100 * d["ci"][1]], [k, k], color=VAR[v][1], linewidth=2)
            ax.plot(100 * d["diff"], k, "o", color=VAR[v][1], markersize=7, markeredgecolor=SURF, markeredgewidth=1.5)
            ax.text(100 * d["ci"][1] + 0.05, k, f"{100 * d['diff']:+.2f} pt", va="center", fontsize=8, color=INK)
            labels.append(f"{s}: {VAR[v][0]}")
            k += 1
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(range(len(labels)), labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Difference in P(first token is a \"go\" word) (FP8 − BF16, points)", fontsize=9, color=INK2)
    ax.set_title("First-token probability (teacher forcing, no sampling)", fontsize=10.5, color=INK, loc="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out / "fig_stopgo.png")


def main():
    m, e, g, out = sys.argv[1:5]
    out = Path(out)
    if m != "-":
        multi(json.loads(Path(m).read_text()), out)
    if e != "-":
        e2e(json.loads(Path(e).read_text()), out)
    if g != "-":
        stopgo(json.loads(Path(g).read_text()), out)


if __name__ == "__main__":
    main()

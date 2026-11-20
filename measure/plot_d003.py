#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# plot_d003.py <multi.json> <e2e.json|-> <stopgo.json|-> <out_dir>
#  の図(図用の venv: matplotlib + japanize-matplotlib)。体裁と色は plot_results.py にそろえる。
#   fig_multi_stream.png : ログごとの連続の時間(中央値の点と最大の線)。10Hz / 1Hz × 4 つの手 / + FP8。1 秒の線つき
#   fig_e2e.png          : 描き直し込みの 1 歩の時間(描き直し + 推論の中央値を積み、最大を線で)。1 秒の線つき
#   fig_stopgo.png       : 止まる場面での「進む」の割合(変種ごと、95% 区間)と、1 トークン目の「進む」系の確率の差
import json
import sys
from pathlib import Path

import japanize_matplotlib  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_results import GRID, INK, INK2, PAL, SURF, VAR, save, style  # noqa: E402

COND = {"pgcv": ("4 つの手(BF16)", PAL[0]), "all": ("+ 本体+出力層 FP8", PAL[2])}


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
            ax.text(0.99, -0.13 - 0.06 * j, f"{name}: 全 {c['n']} 歩 中央値 {c['median']:.0f} / 最大 {c['max']:.0f} ms、"
                    f"1 秒超え {100 * c['over_1s']:.1f}%", transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
        ax.axvline(1000, color=INK2, linestyle="--", linewidth=1)
        ax.text(1000, len(logs) - 0.3, " 1 秒", color=INK2, fontsize=8, va="bottom")
        ax.set_xlim(0, 1150)
        ax.set_title(f"連続 {hz}Hz の間隔(推論だけ。描き直しは除く)", fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("1 歩の時間 ms(点 = 中央値、線の右端 = 最大)", fontsize=9, color=INK2)
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
        ax.barh(i, r, height=0.6, color=PAL[3], edgecolor=SURF, linewidth=1.5, label="描き直し(JPEG の読み込みから)" if i == 0 else None)
        ax.barh(i, e, left=r, height=0.6, color=PAL[0], edgecolor=SURF, linewidth=1.5, label="推論" if i == 0 else None)
        ax.plot([c["max"], c["max"]], [i - 0.3, i + 0.3], color=INK, linewidth=2)
        ax.text(c["max"] + 12, i, f"中央値 {c['median']:.0f} / 最大 {c['max']:.0f} ms、1 秒超え {100 * c['over_1s']:.1f}%",
                va="center", fontsize=8, color=INK)
    names = {"E1": "1Hz", "E10": "10Hz"}

    def label(k):
        hz, cond = names.get(k.split("_")[0], k), COND.get(k.split("_")[1], (k,))[0]
        how = "1 枚ずつ .cpu()(最初の版)" if k.endswith("_cpu") else "pinned にまとめて写す"
        return f"{hz} {cond}\n描き直し: {how}"
    ax.set_yticks(range(len(rows)), [label(k) for k, _ in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.axvline(1000, color=INK2, linestyle="--", linewidth=1)
    ax.set_xlim(0, max(1500, max(c["max"] for _, c in rows) + 420))
    ax.set_xlabel("1 歩の時間 ms(棒 = 中央値の内訳、縦線 = 合計の最大)", fontsize=9, color=INK2)
    ax.set_title("描き直し込みの連続の時間(同じ GPU で続けて動かす)", fontsize=10.5, color=INK, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out / "fig_e2e.png")


def stopgo(G, out):
    S = G["sets"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    ax = axes[0]
    style(ax)
    sets = [("A", "A: 記録で止まっている場面\n(軌跡が 3 s で 2 m 以上 = 進む)", "traj_go"),
            ("B", "B: 判断理由に「止まる」が出る場面\n(判断理由が go の割合)", "cot_go")]
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
    ax.set_xlabel("「進む」の割合 %(点 = 全体、線 = 場面を引き直した 95% 区間)", fontsize=9, color=INK2)
    ax.set_title("止まる場面で「進む」が出る割合", fontsize=10.5, color=INK, loc="left")
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
    ax.set_xlabel("1 トークン目が「進む」系の語になる確率の差(FP8 − BF16、ポイント)", fontsize=9, color=INK2)
    ax.set_title("1 トークン目の確率(teacher forcing、乱数なし)", fontsize=10.5, color=INK, loc="left")
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

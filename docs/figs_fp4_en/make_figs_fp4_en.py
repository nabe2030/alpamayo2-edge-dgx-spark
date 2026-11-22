# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# make_figs_fp4_en.py [out_dir] — English versions of the graphs in the FP4 / TensorRT Edge-LLM article (figures 1-9).
#   The numbers are written directly in this script and are the same as in the Japanese article on Qiita.
#   Numbers, colors and layout are the same as the Japanese version; only the text is in English.
#   Font: Noto Sans CJK JP (the font of the Japanese version, `fonts-noto-cjk` on Ubuntu) if it is found,
#   otherwise DejaVu Sans. Set FIG_FONT=/path/to/font.ttc to use another file.
#   Requires matplotlib only (no data, no model).
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle  # noqa: E402

FONT = os.environ.get("FIG_FONT", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if os.path.exists(FONT):
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
else:
    plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

GRAY, BLUE, ORANGE, TEAL = "#8a8f98", "#2f7fd1", "#d9730d", "#2a9d8f"
T_GRAY, T_BLUE, T_ORANGE, T_TEAL = "#ecedef", "#d6e6f7", "#f8dfc8", "#d3ece8"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SURFACE = "#ffffff"
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)


def style(ax, xmax, xlabel, xmin=0):
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=INK2, labelsize=9, length=0)
    ax.tick_params(axis="y", colors=INK, labelsize=10.5, length=0)


def hbars(ax, ys, vals, colors, labels, xmax, height=0.56):
    ax.barh(ys, vals, color=colors, height=height, edgecolor=SURFACE, linewidth=1.5)
    for y, v, t in zip(ys, vals, labels):
        ax.text(v + xmax * 0.015, y, t, va="center", ha="left", color=INK, fontsize=10.5, bbox=dict(fc=SURFACE, ec="none", pad=1.5))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ---------- Fig 1: headline ----------
rows = ["Official acceleration (tuned)", "+ Four optimizations (previous)", "+ FP8 body (previous)", "+ FP4 body (this work)"]
vals = [1927, 788, 688, 641]
ys = [3, 2, 1, 0]
fig, ax = plt.subplots(figsize=(9, 3.3))
hbars(ax, ys, vals, [GRAY, GRAY, BLUE, ORANGE], ["1,927 ms", "788 ms", "688 ms", "641 ms"], 2300)
style(ax, 2300, "One inference (ms)")
ax.set_yticks(ys)
ax.set_yticklabels(rows)
ax.set_ylim(-0.6, 3.6)
handles = [Patch(color=GRAY, label="16-bit (BF16)"), Patch(color=BLUE, label="8-bit (FP8)"), Patch(color=ORANGE, label="4-bit (FP4)")]
ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9.5)
fig.tight_layout()
save(fig, "fig01_headline.png")

# ---------- Fig 2: pipeline matrix ----------
stages = ["ViT\nreads 22 images", "Prefill\nreads the whole input", "Reasoning text\none token at a time", "Expert\noutputs the trajectory"]
configs = [
    ("PyTorch  BF16\n(four optimizations)", ["16-bit", "16-bit", "16-bit", "16-bit"]),
    ("PyTorch  FP4 body\n(custom kernel)", ["16-bit", "16-bit", "4-bit\n(weights only)", "16-bit"]),
    ("TensorRT  NVFP4 W4A4", ["16-bit", "4-bit (weights and\nactivations)", "4-bit (weights and\nactivations)", "not supported"]),
    ("TensorRT  NVFP4 W4A16", ["16-bit", "4-bit\n(weights only)", "4-bit\n(weights only)", "not supported"]),
]
LW, CW, RH, HH = 3.4, 2.55, 0.95, 1.0
fig, ax = plt.subplots(figsize=(11.5, 4.9))
ax.set_xlim(0, LW + CW * 4 + 0.1)
ax.set_ylim(-0.1, HH + RH * 4 + 0.35)
ax.axis("off")
top = HH + RH * 4 + 0.2
for j, s in enumerate(stages):
    x0 = LW + CW * j
    ax.add_patch(FancyBboxPatch((x0 + 0.12, top - HH + 0.08), CW - 0.24, HH - 0.2, boxstyle="round,pad=0.02,rounding_size=0.08", fc=SURFACE, ec=INK2, lw=1.1))
    ax.text(x0 + CW / 2, top - HH / 2 - 0.02, s, ha="center", va="center", fontsize=10.5, color=INK, linespacing=1.5)
    if j < 3:
        ax.add_patch(FancyArrowPatch((x0 + CW - 0.12, top - HH / 2 - 0.02), (x0 + CW + 0.12, top - HH / 2 - 0.02), arrowstyle="-|>", mutation_scale=12, color=INK2, lw=1.1))
for i, (name, cells) in enumerate(configs):
    y0 = top - HH - RH * (i + 1)
    ax.text(LW - 0.2, y0 + RH / 2, name, ha="right", va="center", fontsize=10.5, color=INK, linespacing=1.5)
    for j, c in enumerate(cells):
        x0 = LW + CW * j
        if c.startswith("4"):
            fc, ec, ls, tc = T_ORANGE, ORANGE, "-", INK
        elif c.startswith("16"):
            fc, ec, ls, tc = T_GRAY, GRAY, "-", INK
        else:
            fc, ec, ls, tc = SURFACE, GRAY, (0, (4, 3)), INK2
        ax.add_patch(Rectangle((x0 + 0.12, y0 + 0.1), CW - 0.24, RH - 0.2, fc=fc, ec=ec, lw=1.3, ls=ls))
        ax.text(x0 + CW / 2, y0 + RH / 2, c, ha="center", va="center", fontsize=10.5, color=tc, linespacing=1.4)
fig.tight_layout(pad=0.4)
save(fig, "fig02_pipeline.png")

# ---------- Fig 3: FP4 weight format ----------
fig, ax = plt.subplots(figsize=(11, 2.9))
ax.set_xlim(0, 22)
ax.set_ylim(0, 5.2)
ax.axis("off")
S = 0.5
for b in range(2):
    xb = 0.6 + b * 9.2
    for k in range(16):
        ax.add_patch(Rectangle((xb + k * S, 2.6), S, S * 1.5, fc=T_ORANGE, ec=ORANGE, lw=1.0))
    ax.text(xb + 8 * S, 3.75, "16 weights (4 bits each)", ha="center", va="bottom", fontsize=10.5, color=INK)
    ax.add_patch(Rectangle((xb + 2.5 * S, 0.9), 11 * S, 0.9, fc=T_BLUE, ec=BLUE, lw=1.1))
    ax.text(xb + 8 * S, 1.35, "Scale of these 16 (8 bits)", ha="center", va="center", fontsize=10, color=INK)
    ax.add_patch(FancyArrowPatch((xb + 8 * S, 1.85), (xb + 8 * S, 2.55), arrowstyle="-|>", mutation_scale=11, color=INK2, lw=1.0))
ax.text(18.55, 2.6 + S * 0.75, "…", ha="center", va="center", fontsize=16, color=INK2)
ax.add_patch(Rectangle((19.2, 2.6), 2.5, S * 1.5, fc=T_GRAY, ec=GRAY, lw=1.1))
ax.text(20.45, 2.6 + S * 0.75, "Row scale", ha="center", va="center", fontsize=10, color=INK)
ax.text(20.45, 3.75, "One per row (FP32)", ha="center", va="bottom", fontsize=10.5, color=INK)
ax.text(0.6, 4.75, "One row of the weight matrix", ha="left", va="center", fontsize=11, color=INK2)
fig.tight_layout(pad=0.3)
save(fig, "fig03_format.png")

# ---------- Fig 4: decode per token / 10 Hz continuous ----------
rows = ["BF16\n(four optimizations)", "FP8 body\n(custom kernel)", "FP4 body\n(custom kernel, GPTQ)"]
cols = [GRAY, BLUE, ORANGE]
ys = [2, 1, 0]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), sharey=True)
hbars(axes[0], ys, [16.6, 11.0, 8.47], cols, ["16.6 ms", "11.0 ms", "8.47 ms"], 20.5)
style(axes[0], 20.5, "ms")
axes[0].set_title("Reasoning text, per token", loc="left", fontsize=12, color=INK, pad=10)
hbars(axes[1], ys, [643, 536, 475], cols, ["643 ms (max 870)", "536 ms (max 692)", "475 ms (max 609)"], 1000)
style(axes[1], 1000, "ms")
axes[1].set_title("Continuous inference on 10 Hz input (median)", loc="left", fontsize=12, color=INK, pad=10)
axes[0].set_yticks(ys)
axes[0].set_yticklabels(rows)
axes[0].set_ylim(-0.6, 2.6)
fig.tight_layout(w_pad=3)
save(fig, "fig04_decode.png")

# ---------- Fig 5: rounding methods ----------
rows = ["FP8 body (reference)", "FP4 plain rounding (RTN)", "FP4 scale search", "FP4 GPTQ (adopted)", "FP4 GPTQ + scale search"]
cols = [BLUE, ORANGE, ORANGE, ORANGE, ORANGE]
ys = [4, 3, 2, 1, 0]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
hbars(axes[0], ys, [1.4, 5.0, 4.1, 1.5, 2.2], cols, ["1.4 %", "5.0 %", "4.1 %", "1.5 %", "2.2 %"], 6.2)
style(axes[0], 6.2, "% (smaller = closer to BF16)")
axes[0].set_title("Trajectory difference (% of seed-to-seed spread)", loc="left", fontsize=12, color=INK, pad=10)
hbars(axes[1], ys, [91.7, 66.7, 75.8, 85.8, 85.0], cols, ["91.7 %", "66.7 %", "75.8 %", "85.8 %", "85.0 %"], 115)
style(axes[1], 115, "% (larger = closer to BF16)")
axes[1].set_xticks([0, 20, 40, 60, 80, 100])
axes[1].set_title("Reasoning text identical to BF16", loc="left", fontsize=12, color=INK, pad=10)
axes[0].set_yticks(ys)
axes[0].set_yticklabels(rows)
axes[0].set_ylim(-0.6, 4.6)
fig.tight_layout(w_pad=3)
save(fig, "fig05_rounding.png")

# ---------- Fig 6: first-token shift ----------
rows = ["FP8 body", "FP4 body, plain rounding (RTN)", "FP4 body, GPTQ (adopted)", "FP4 body, GPTQ + FP8 output layer"]
vals = [0.04, -2.14, -0.47, 0.37]
lo = [-0.10, -2.65, -0.64, 0.18]
hi = [0.19, -1.66, -0.30, 0.56]
labs = ["+0.04", "−2.14", "−0.47", "+0.37"]
cols = [BLUE, ORANGE, ORANGE, ORANGE]
ys = [3, 2, 1, 0]
fig, ax = plt.subplots(figsize=(9.5, 3.5))
ax.barh(ys, vals, color=cols, height=0.52, edgecolor=SURFACE, linewidth=1.5)
for y, v, a, b, t in zip(ys, vals, lo, hi, labs):
    ax.plot([a, b], [y, y], color=INK, lw=1.3, solid_capstyle="butt")
    ax.plot([a, a], [y - 0.1, y + 0.1], color=INK, lw=1.3)
    ax.plot([b, b], [y - 0.1, y + 0.1], color=INK, lw=1.3)
    if v < 0:
        ax.text(a - 0.08, y, t, va="center", ha="right", color=INK, fontsize=10.5, bbox=dict(fc=SURFACE, ec="none", pad=1.5))
    else:
        ax.text(b + 0.08, y, t, va="center", ha="left", color=INK, fontsize=10.5, bbox=dict(fc=SURFACE, ec="none", pad=1.5))
ax.axvline(0, color=INK2, lw=1.0)
style(ax, 1.4, "Change in the probability that the reasoning text starts with a \"go\" word (points)", xmin=-3.4)
ax.set_yticks(ys)
ax.set_yticklabels(rows)
ax.set_ylim(-0.6, 3.9)
ax.text(-3.3, 3.62, "← toward \"stop\"", ha="left", va="center", fontsize=10, color=INK2, bbox=dict(fc=SURFACE, ec="none", pad=2))
ax.text(1.35, 3.62, "toward \"go\" →", ha="right", va="center", fontsize=10, color=INK2, bbox=dict(fc=SURFACE, ec="none", pad=2))
fig.tight_layout()
save(fig, "fig06_firsttoken.png")

# ---------- Fig 7: stages ----------
rows = ["PyTorch  BF16", "PyTorch  FP8 body (custom)", "PyTorch  FP4 body (custom)", "TensorRT  FP16", "TensorRT  NVFP4 W4A4", "TensorRT  NVFP4 W4A16"]
cols = [GRAY, BLUE, ORANGE, GRAY, ORANGE, ORANGE]
ys = [6, 5, 4, 2, 1, 0]
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3), sharey=True)
hbars(axes[0], ys, [241, 243, 241, 187, 188, 210], cols, ["241", "243", "241", "187", "188", "210"], 300, height=0.6)
style(axes[0], 300, "ms")
axes[0].set_title("ViT (22 images)", loc="left", fontsize=12, color=INK, pad=10)
hbars(axes[1], ys, [176, 178, 175, 161, 104, 184], cols, ["176", "178", "175", "161", "104", "184"], 230, height=0.6)
style(axes[1], 230, "ms")
axes[1].set_title("Prefill (about 3,260 tokens)", loc="left", fontsize=12, color=INK, pad=10)
hbars(axes[2], ys, [16.6, 11.0, 8.47, 17.3, 9.98, 9.83], cols, ["16.6", "11.0", "8.47", "17.3", "9.98", "9.83"], 21.5, height=0.6)
style(axes[2], 21.5, "ms")
axes[2].set_title("Reasoning text, per token", loc="left", fontsize=12, color=INK, pad=10)
axes[0].set_yticks(ys)
axes[0].set_yticklabels(rows)
axes[0].set_ylim(-0.7, 6.7)
handles = [Patch(color=GRAY, label="16-bit (BF16 / FP16)"), Patch(color=BLUE, label="8-bit (FP8)"), Patch(color=ORANGE, label="4-bit (FP4 / NVFP4)")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.55, -0.01))
fig.tight_layout(w_pad=2.5, rect=(0, 0.07, 1, 1))
save(fig, "fig07_stages.png")

# ---------- Fig 8: greedy agreement ----------
rows = ["TensorRT  FP16", "TensorRT  NVFP4 W4A4", "TensorRT  NVFP4 W4A16", "PyTorch  FP4 GPTQ\n(4-bit in decode only)", "PyTorch  FP4 plain rounding\n(4-bit in prefill too)"]
vals = [18, 10, 13, 18, 14]
cols = [GRAY, ORANGE, ORANGE, ORANGE, ORANGE]
ys = [5, 4, 3, 1, 0]
fig, ax = plt.subplots(figsize=(9, 3.9))
hbars(ax, ys, vals, cols, ["18 / 20", "10 / 20", "13 / 20", "18 / 20", "14 / 20"], 21.6)
style(ax, 21.6, "Scenes whose reasoning text exactly matches PyTorch BF16 (out of 20)")
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_yticks(ys)
ax.set_yticklabels(rows)
ax.set_ylim(-0.7, 5.7)
fig.tight_layout()
save(fig, "fig08_greedy.png")

# ---------- Fig 9: concat aliasing bug ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
W, H = 1.0, 0.62


def cell(ax, x, y, w, text, fc, ec):
    ax.add_patch(Rectangle((x, y), w, H, fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + H / 2, text, ha="center", va="center", fontsize=10.5, color=INK)


for ax in axes:
    ax.set_xlim(-1.3, 4.4)
    ax.set_ylim(-0.8, 3.4)
    ax.axis("off")
ax = axes[0]
ax.set_title("Intended layout (one row = one token)", loc="left", fontsize=12, color=INK, pad=8)
for r in range(4):
    y = (3 - r) * 0.85
    ax.text(-0.15, y + H / 2, f"row {r}", ha="right", va="center", fontsize=10.5, color=INK2)
    cell(ax, 0, y, 2 * W, f"q (token {r})", T_BLUE, BLUE)
    cell(ax, 2 * W, y, W, f"k ({r})", T_TEAL, TEAL)
    cell(ax, 3 * W, y, W, f"v ({r})", T_GRAY, GRAY)
ax.text(1.0, -0.45, "q columns", ha="center", va="center", fontsize=10, color=INK2)
ax.text(2.5, -0.45, "k columns", ha="center", va="center", fontsize=10, color=INK2)
ax.text(3.5, -0.45, "v columns", ha="center", va="center", fontsize=10, color=INK2)
ax = axes[1]
ax.set_title("Layout actually written (2 or more rows)", loc="left", fontsize=12, color=INK, pad=8)
for r in range(4):
    y = (3 - r) * 0.85
    ax.text(-0.15, y + H / 2, f"row {r}", ha="right", va="center", fontsize=10.5, color=INK2)
    if r < 2:
        cell(ax, 0, y, 2 * W, f"q (token {2 * r})", T_BLUE, BLUE)
        cell(ax, 2 * W, y, 2 * W, f"q (token {2 * r + 1})", T_BLUE, BLUE)
    else:
        ax.add_patch(Rectangle((0, y), 4 * W, H, fc=SURFACE, ec=GRAY, lw=1.2, ls=(0, (4, 3)), hatch="///"))
        ax.text(2 * W, y + H / 2, "no correct values", ha="center", va="center", fontsize=10.5, color=INK, bbox=dict(fc=SURFACE, ec="none", pad=2.5))
ax.plot([2 * W, 2 * W], [2 * 0.85 - 0.1, 3 * 0.85 + H + 0.1], color=INK, lw=1.2, ls=(0, (2, 2)))
ax.text(1.0, -0.45, "q columns", ha="center", va="center", fontsize=10, color=INK2)
ax.text(3.0, -0.45, "columns read as k and v", ha="center", va="center", fontsize=10, color=INK2)
fig.tight_layout(w_pad=1.5)
save(fig, "fig09_bug.png")
print("ok")

"""English versions of the article graphs (figures 1-5, 8-10).
Numbers are identical to the Japanese versions (make_figs.py / make_figs_v2.py)."""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge, FancyBboxPatch, FancyArrowPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else "figs_en/"
os.makedirs(OUT, exist_ok=True)
SRC = "Data: Argoverse 2 Sensor Dataset (© 2021 Argo AI, LLC, CC BY-NC-SA 4.0) / NVIDIA DGX Spark (GB10)"
C_BF = "#8a8f98"; C_F8 = "#2f7fd1"; C_F8H = "#2a9d8f"; C_RED = "#d1432f"; C_GRN = "#3a9a5c"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})


def foot(fig, y=0.005):
    fig.text(0.5, y, SRC, ha="center", va="bottom", fontsize=8.5, color="#555")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------- Figure 1: speed steps ----------
def fig01():
    conds = ["Official, no acceleration", "Official compile", "Official tuned",
             "+ Load the processor once", "+ Resize images on the GPU", "+ Compile the ViT too",
             "+ FP8 body only", "+ FP8 body + output layer"]
    data = np.array([
        [1076, 318, 314, 502, 313],
        [1075, 315, 176, 389, 52],
        [1081, 320, 175, 296, 52],
        [82, 318, 176, 299, 53],
        [17, 318, 176, 298, 53],
        [17, 241, 176, 299, 52],
        [18, 243, 178, 199, 54],
        [18, 243, 179, 180, 54],
    ])
    e2e = [2524, 2011, 1927, 932, 865, 790, 695, 678]
    names = ["Preprocessing", "ViT", "Prefill", "Reasoning decode", "Trajectory (diffusion)"]
    cols = ["#e8926b", "#8fbfe8", "#5f9fd6", "#2f6fb5", "#6fbf86"]
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    y = np.arange(len(conds))[::-1]
    left = np.zeros(len(conds))
    for j in range(5):
        ax.barh(y, data[:, j], left=left, color=cols[j], label=names[j], height=0.62, edgecolor="white")
        left += data[:, j]
    for yi, v in zip(y, e2e):
        ax.text((1030 if 760 <= v < 1000 else v + 30), yi, f"{v} ms", va="center", fontsize=11, fontweight="bold")
    ax.axvline(1000, color=C_RED, ls="--", lw=1.6)
    ax.text(1010, len(conds) - 0.45, "1 s (the 1 Hz deadline)", color=C_RED, fontsize=10.5, va="bottom", ha="left")
    ax.set_yticks(y); ax.set_yticklabels(conds, fontsize=11.5)
    ax.set_xlim(0, 2950); ax.set_ylim(-0.6, len(conds) + 0.05)
    ax.set_xlabel("Time for one inference (ms, median of 20 runs)")
    ax.set_title("One-inference time as each optimization is added (scene with 18 reasoning tokens)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", ncol=1, frameon=False, fontsize=10.5)
    fig.subplots_adjust(bottom=0.16)
    foot(fig, 0.01)
    save(fig, "fig01_speed_steps.png")


# ---------- Figure 2: pipeline ----------
def fig02():
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.8); ax.axis("off")
    boxes = [
        ("6 cameras\n22 images\n+ ego history\n+ nav text", "", "#eef1f5"),
        ("Preprocess\n(resize,\nnormalize)", "1081 → 18 ms", "#fde9d9"),
        ("Image\nencoder\n(ViT)", "320 → 243 ms", "#e3f0fb"),
        ("Language\nmodel\nprefill", "175 → 178 ms", "#e3f0fb"),
        ("Language\nmodel decodes\nreasoning,\none token\nat a time", "296 → 199 ms", "#d5e8fa"),
        ("Action\nexpert\n(10 diffusion\nsteps)", "52 → 54 ms", "#e6f4ea"),
        ("Output\nreasoning text\n+ 64-point\ntrajectory\n(6.4 s ahead)", "", "#eef1f5"),
    ]
    w = 1.42; gap = 0.31; x = 0.12; y0 = 1.75; h = 1.9
    for i, (t, s, c) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=c, ec="#667", lw=1.2))
        ax.text(x + w / 2, y0 + h / 2, t, ha="center", va="center", fontsize=10, linespacing=1.35)
        if s:
            ax.text(x + w / 2, y0 - 0.38, s, ha="center", va="center", fontsize=11,
                    color=C_RED if i == 1 else "#223", fontweight="bold")
        if i < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.03, y0 + h / 2), (x + w + gap - 0.03, y0 + h / 2),
                                         arrowstyle="-|>", mutation_scale=14, color="#445", lw=1.3))
        x += w + gap
    ax.text(6, 4.45, "One inference of Alpamayo 2 Edge", ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(6, 0.72, "Numbers under the boxes: official tuned → this article's setup (four optimizations + FP8 body only).\n"
                     "Single-inference measurement, example with 18 reasoning tokens.",
            ha="center", va="center", fontsize=10.5, color="#333", linespacing=1.4)
    foot(fig, 0.01)
    save(fig, "fig02_pipeline.png")


# ---------- Figure 3: tokens ----------
def fig03():
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    n = np.arange(0, 81)
    lines = [("BF16", 417, 15.43, C_BF, 37), ("FP8 body only (recommended)", 419, 9.98, C_F8, 58),
             ("FP8 body + output layer (reference)", 417, 9.23, C_F8H, 63)]
    for lab, a, b, c, lim in lines:
        ax.plot(n, a + b * n, color=c, lw=2.6, label=f"{lab}: {a} + {b} × tokens")
        ax.plot([lim], [a + b * lim], "o", color=c, ms=8, mec="#333", mew=0.6)
    ax.axhline(1000, color=C_RED, ls="--", lw=1.6)
    ax.text(79.5, 1012, "1 s", color=C_RED, fontsize=11, va="bottom", ha="right")
    ax.axvspan(0, 37, color="#f3e7b3", alpha=0.45, lw=0)
    ax.text(18.5, 290, "Lengths actually seen in the 12 logs\n(longest: 37 tokens)", ha="center", va="center",
            fontsize=10.5, color="#6a5a10")
    ax.annotate("BF16: up to 37 tokens", xy=(37, 988), xytext=(10, 1190), fontsize=11, color="#444",
                arrowprops=dict(arrowstyle="-", color="#666"))
    ax.annotate("FP8 body only: up to 58 tokens", xy=(58, 998), xytext=(37, 690), fontsize=11, color=C_F8,
                arrowprops=dict(arrowstyle="-", color=C_F8))
    ax.annotate("FP8 body + output layer:\nup to 63 tokens", xy=(63, 998), xytext=(56, 480), fontsize=11, color="#1f7a70",
                arrowprops=dict(arrowstyle="-", color="#1f7a70"))
    ax.set_xlim(0, 80); ax.set_ylim(200, 1700)
    ax.set_xlabel("Length of the reasoning text (tokens)"); ax.set_ylabel("Time for one inference (ms)")
    ax.set_title("Inference time is set by the reasoning length (input every 1 s, fit to 12 logs)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    fig.subplots_adjust(bottom=0.15)
    foot(fig, 0.01)
    save(fig, "fig03_tokens.png")


# ---------- Figure 4: camera ----------
def fig04():
    fig, axs = plt.subplots(1, 2, figsize=(12, 6.2))

    def draw(ax, cams, title, note):
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.45, 1.6); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=13.5, fontweight="bold", pad=6)
        for yaw, fov, r, col, lab, lr, la in cams:
            a = 90 + yaw
            ax.add_patch(Wedge((0, 0), r, a - fov / 2, a + fov / 2, fc=col, ec="#334", lw=1.0, alpha=0.55))
            ang = np.deg2rad(a + la)
            ax.text(lr * np.cos(ang), lr * np.sin(ang), lab, ha="center", va="center", fontsize=9.5)
        ax.add_patch(FancyBboxPatch((-0.07, -0.16), 0.14, 0.32, boxstyle="round,pad=0.01,rounding_size=0.04",
                                    fc="#333", ec="#111"))
        ax.annotate("", xy=(0, 1.5), xytext=(0, 1.36), arrowprops=dict(arrowstyle="-|>", color="#333"))
        ax.text(0.07, 1.47, "forward", fontsize=9.5, va="center")
        ax.text(0, -1.38, note, ha="center", va="center", fontsize=10.5, color="#333")

    av2 = [
        (0, 47, 1.0, "#f2b36b", "front center\n47°", 1.19, 0),
        (45, 62.5, 0.9, "#9cc3ea", "front left\n62.5°", 1.16, 0),
        (-45, 62.5, 0.9, "#9cc3ea", "front right\n62.5°", 1.16, 0),
        (99, 62.5, 0.9, "#b9dcc2", "side left\n62.5°", 1.2, 0),
        (-99, 62.5, 0.9, "#b9dcc2", "side right\n62.5°", 1.2, 0),
        (153, 62.5, 0.9, "#d7c3e8", "rear left\n62.5°", 1.14, 0),
        (-153, 62.5, 0.9, "#d7c3e8", "rear right\n62.5°", 1.14, 0),
    ]
    slots = [
        (0, 120, 0.82, "#9cc3ea", "front wide\n120°", 1.06, 40),
        (0, 30, 1.18, "#f2b36b", "tele\n30°", 1.1, -27),
        (90, 120, 0.62, "#b9dcc2", "left\n120°", 0.82, 0),
        (-90, 120, 0.62, "#b9dcc2", "right\n120°", 0.82, 0),
        (150, 70, 0.95, "#d7c3e8", "rear left\n70°", 1.18, 0),
        (-150, 70, 0.95, "#d7c3e8", "rear right\n70°", 1.18, 0),
    ]
    draw(axs[0], av2, "Argoverse 2: seven ring cameras", "No 120° camera and no telephoto camera")
    draw(axs[1], slots, "The six camera slots the model expects",
         "AV2 images are re-rendered into virtual cameras with these FOVs")
    fig.suptitle("Camera fields of view (top view)", fontsize=15, fontweight="bold", y=0.99)
    foot(fig, 0.0)
    save(fig, "fig04_camera.png")


# ---------- Figure 5: input ----------
def fig05():
    fig, axs = plt.subplots(1, 2, figsize=(12.6, 5.6))
    ax = axs[0]
    labs = ["Re-render 120°\n+ tele crop\n(used)", "AV2 cameras\nas is", "Re-render 120°\n+ tele = front\nimage",
            "Reference:\nkeep the last\nmotion"]
    v = [2.58, 3.67, 3.99, 5.89]
    cols = [C_GRN, C_BF, C_BF, "#cfd3d8"]
    ax.bar(range(4), v, color=cols, width=0.62)
    for i, val in enumerate(v):
        ax.text(i, val + 0.1, f"{val:.2f} m", ha="center", va="bottom", fontsize=11.5, fontweight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=9.5)
    ax.set_ylim(0, 6.9); ax.set_ylabel("ADE vs. the recorded drive (m, lower is better)")
    ax.set_title("How the input is assembled\n(30 scenes × 5 seeds)", fontsize=12.5, fontweight="bold")
    ax = axs[1]
    labs2 = ["Left turn\n(63)", "Right turn\n(81)", "Straight\n(157)", "Other\n(80)", "Stopped\n(19)"]
    v2 = [1.61, 2.65, 0.15, 0.16, 0.00]
    ax.bar(range(5), v2, color=["#d98a4b", "#d98a4b", C_BF, C_BF, C_BF], width=0.62)
    for i, val in enumerate(v2):
        ax.text(i, val + 0.06, f"+{val:.2f} m", ha="center", va="bottom", fontsize=11.5, fontweight="bold")
    ax.set_xticks(range(5)); ax.set_xticklabels(labs2, fontsize=10.5)
    ax.set_ylim(0, 3.2); ax.set_ylabel("Increase in ADE (m)")
    ax.set_title("Nav text always \"Continue straight\"\n(400 scenes; number of scenes in parentheses)",
                 fontsize=12.5, fontweight="bold")
    fig.suptitle("How much the input construction changes the result (BF16)", fontsize=14.5, fontweight="bold", y=1.03)
    fig.subplots_adjust(bottom=0.24, wspace=0.26)
    foot(fig, 0.01)
    save(fig, "fig05_input.png")


# ---------- Figure 8: stream ----------
def fig08():
    groups = ["Inference only\ninput every 0.1 s", "Inference only\ninput every 1 s",
              "Incl. JPEG decode\nand re-rendering\ninput every 1 s"]
    series = [
        ("Four optimizations (BF16)", C_BF, [641, 757, 849], [867, 977, 1077]),
        ("+ FP8 body only (recommended)", C_F8, [528, 638, 736], [685, 780, 878]),
        ("+ FP8 body + output layer (reference)", C_F8H, [512, 620, 715], [644, 752, 852]),
    ]
    over = ["BF16 0% / both FP8 setups 0%", "BF16 0% / both FP8 setups 0%", "BF16 1.4% / both FP8 setups 0%"]
    fig, ax = plt.subplots(figsize=(12, 6.2))
    x = np.arange(3); w = 0.26
    for k, (lab, col, med, mx) in enumerate(series):
        xs = x + (k - 1) * w
        ax.bar(xs, med, w * 0.92, color=col, label=lab)
        for xi, md, m in zip(xs, med, mx):
            ax.plot([xi, xi], [md, m], color="#222", lw=1.5)
            ax.plot([xi - 0.05, xi + 0.05], [m, m], color="#222", lw=1.5)
            ax.text(xi, md / 2, f"{md}", ha="center", va="center", color="white", fontsize=11, fontweight="bold")
            ytxt = m + 12
            if 940 < ytxt < 1040:
                ytxt = 1040
            ax.text(xi, ytxt, f"{m}", ha="center", va="bottom", fontsize=9.5, color="#222")
    ax.axhline(1000, color=C_RED, ls="--", lw=1.6, zorder=0)
    ax.text(-0.56, 1008, "1 s", color=C_RED, fontsize=11, va="bottom", ha="left")
    for xi, t in zip(x, over):
        ax.text(xi, 1235, "Share of runs over 1 s\n" + t, ha="center", va="center", fontsize=10,
                color=C_RED if "1.4%" in t else "#333")
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=11.5)
    ax.set_ylim(0, 1330); ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("Time per inference (ms)")
    ax.set_title("12 driving logs run continuously (number in bar = median, above whisker = max)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=10.5)
    fig.subplots_adjust(bottom=0.3)
    foot(fig, 0.01)
    save(fig, "fig08_stream.png")


# ---------- Figure 9: quality ----------
def fig09():
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.4))
    ax = axs[0]
    labs = ["BF16 vs. BF16\n(different seed)", "FP8 body only\n(same seed)", "FP8 body + output\nlayer (same seed)"]
    v = [2.302, 0.041, 0.045]; cols = [C_BF, C_F8, C_F8H]
    ax.bar(range(3), v, color=cols, width=0.6)
    ax.set_yscale("log"); ax.set_ylim(0.01, 12)
    for i, (val, t) in enumerate(zip(v, ["2.30 m", "0.041 m\n(1.8%)", "0.045 m\n(2.0%)"])):
        ax.text(i, val * 1.25, t, ha="center", va="bottom", fontsize=11.5, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=10.5)
    ax.set_ylabel("Trajectory difference (m, log scale)")
    ax.set_title("Trajectory difference", fontsize=13, fontweight="bold")
    ax = axs[1]
    v = [23.3, 91.2, 87.2]
    ax.bar(range(3), v, color=cols, width=0.6)
    for i, val in enumerate(v):
        ax.text(i, val + 1.5, f"{val}%", ha="center", va="bottom", fontsize=11.5, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=10.5)
    ax.set_ylim(0, 108); ax.set_ylabel("Runs with identical reasoning text (%)")
    ax.set_title("Reasoning text match", fontsize=13, fontweight="bold")
    fig.suptitle("Output difference between BF16 and FP8 (400 scenes × 5 seeds)", fontsize=14.5, fontweight="bold", y=1.0)
    fig.subplots_adjust(bottom=0.2, wspace=0.28)
    foot(fig, 0.01)
    save(fig, "fig09_quality.png")


# ---------- Figure 10: stop/go ----------
def fig10():
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 5.4))
    ax = axs[0]
    labs = ["BF16", "FP8 body\nonly", "FP8 body +\noutput layer"]
    v = [26.3, 26.3, 26.2]; lo = [14.3, 14.3, 14.3]; hi = [40.3, 40.4, 40.2]
    cols = [C_BF, C_F8, C_F8H]
    ax.bar(range(3), v, color=cols, width=0.6,
           yerr=[np.array(v) - np.array(lo), np.array(hi) - np.array(v)], capsize=6, ecolor="#333")
    for i, val in enumerate(v):
        ax.text(i, hi[i] + 1.2, f"{val}%", ha="center", va="bottom", fontsize=11.5, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=11)
    ax.set_ylim(0, 50); ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("Runs where the trajectory starts moving (%)")
    ax.set_title("Trajectory: share that starts moving\nin stopped scenes (unchanged)", fontsize=12.5, fontweight="bold")
    ax = axs[1]
    labs2 = ["FP8 body\nonly", "FP8 output\nlayer only", "FP8 body +\noutput layer"]
    v = [0.37, 0.85, 1.45]; lo = [0.03, 0.50, 0.90]; hi = [0.73, 1.25, 2.07]
    ax.bar(range(3), v, color=[C_F8, "#c9a227", C_F8H], width=0.6,
           yerr=[np.array(v) - np.array(lo), np.array(hi) - np.array(v)], capsize=6, ecolor="#333")
    for i, val in enumerate(v):
        ax.text(i, hi[i] + 0.06, f"+{val} pt", ha="center", va="bottom", fontsize=11.5, fontweight="bold")
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(labs2, fontsize=11)
    ax.set_ylim(-0.2, 2.5); ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("Change in P(first token is a \"go\" word)\n(pt, vs. BF16)")
    ax.set_title("Start of the reasoning: shift toward \"go\"\n(appears when the output layer is FP8)",
                 fontsize=12.5, fontweight="bold")
    fig.suptitle("22 stopped scenes × 64 seeds (whiskers: 95% interval)", fontsize=14.5, fontweight="bold", y=1.03)
    fig.subplots_adjust(bottom=0.2, wspace=0.32)
    foot(fig, 0.01)
    save(fig, "fig10_stopgo.png")


for f in (fig01, fig02, fig03, fig04, fig05, fig08, fig09, fig10):
    f()
print("done")

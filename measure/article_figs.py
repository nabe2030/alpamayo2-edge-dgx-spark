#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# article_figs.py <out_dir> [fig6|fig7|fig11 ...] — redraws the article figures that contain AV2 images (figures 6, 7 and 11)
# from the recorded runs.
#   Figure 6 : left turn (log 9e9bcfb7, t0 = 8.5 s). Re-rendered input (wide_ft) vs. AV2 cameras as is (nat).
#              Uses the records of the input-layout comparison (raw/cfg.jsonl and the inputs in raw/samples).
#   Figure 7 : one scene (log 02678d04, t0 = 5.0 s) as input built in 3 layouts, drawn with av2_build.render.
#              The front 3 slots for each layout, plus the other 3 slots of the adopted layout (wide_ft).
#   Figure 11: pedestrian crossing (v_d46ff5df_8.5). Uses the records in raw/wide/wide.jsonl
#              (BF16, FP8 body only and FP8 body + output layer, 5 seeds each). Front image and top views only;
#              the reasoning per seed is in the article text.
#   The reasoning text is the model output as is. An attribution line (Argoverse 2, © 2021 Argo AI, LLC,
#   CC BY-NC-SA 4.0, modified) is added below each figure. Width 1600 px and body text of 26 px or more,
#   so that the figures stay readable when scaled down to about 800 px.
#
# Usage (after sourcing env.sh; PYTHONPATH must include av2_input/ and measure/):
#   $PY article_figs.py <out_dir>        # -> <out_dir>/fig06_left_turn.jpg, fig07_input_layouts.jpg, fig11_pedestrian_stop.jpg
#   Also writes <out_dir>/<fig>_src.png (the PNG before JPEG compression). JPEG quality 90, 4:4:4.
# Environment: A2E_WORK = where the records and data are (default: one level above this file).
#              A2E_FIG_DEVICE = device for re-rendering (default cpu).
# Records read (under A2E_WORK):
#   Figure 6 : raw/cfg.jsonl (records of the input-layout comparison written by av2_eval.py; kind=scene rows hold the
#              recorded trajectory, kind=run rows hold the trajectory, ADE and reasoning of each seed) and
#              raw/samples/<first 8 chars of the log>_<t0>_<layout>.pt (the input at that time: image_frames, camera_indices)
#   Figure 7 : data/av2/val/02678d04... (the AV2 log, fetched with av2_fetch.py)
#   Figure 11: raw/wide/wide.jsonl and raw/wide/scenes_t0.csv (records of measure/run_wide.sh) and the log
# Fonts: Noto Sans CJK (/usr/share/fonts/opentype/noto/; fonts-noto-cjk on Ubuntu).
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import av2_build as B  # noqa: E402
from av2_fig import ORIGIN_H, project  # noqa: E402

N = Path(__import__("os").environ.get("A2E_WORK") or HERE.parent)
# Device for re-rendering. CPU by default (does not disturb other work on the GPU). A2E_FIG_DEVICE=cuda for the GPU
DEV = __import__("os").environ.get("A2E_FIG_DEVICE", "cpu")

WID = 1600
BG = (252, 252, 251)
INK = (11, 11, 11)
SUB = (82, 81, 78)
GT = (0, 131, 0)
COL = {"wide_ft": (214, 40, 40), "nat": (42, 100, 214),
       # The 3 configurations in figure 11: same colors as the article graphs. BF16 = #8a8f98, FP8 body only = #2f7fd1, FP8 body + output layer = #2a9d8f
       "base": (0x8a, 0x8f, 0x98), "fp8body": (0x2f, 0x7f, 0xd1), "fp8all": (0x2a, 0x9d, 0x8f)}
# Text colors (the BF16 gray is too light for text, so a darker gray is used)
TCOL = {**COL, "base": (0x5f, 0x64, 0x6c)}
NAME = {"base": "BF16", "fp8body": "FP8 body only", "fp8all": "FP8 body + output layer"}
CREDIT = ("Images: Argoverse 2 Sensor Dataset (© 2021 Argo AI, LLC), CC BY-NC-SA 4.0 "
           "(https://creativecommons.org/licenses/by-nc-sa/4.0/). The original camera images were modified "
           "(re-rendered to virtual cameras, cropped and resized, trajectories and text overlaid).")


def font(size, bold=False):
    cands = (["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"] if bold else []) + [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf"]
    for p in cands:
        if Path(p).exists():
            return ImageFont.truetype(p, size, index=0)  # index 0 of the .ttc = JP
    raise SystemExit("Noto Sans CJK font not found")


def wrap(dr, text, f, width):
    """Wrap text to fit in width (English by word, CJK by character)."""
    import re
    out, cur = [], ""
    # an ASCII word is one unit, whitespace separates, anything else (CJK) is one character each
    for t in re.findall(r"[\x21-\x7e]+|\s+|.", text):
        cand = cur + t
        if dr.textlength(cand, font=f) <= width or not cur.strip():
            cur = cand
        else:
            out.append(cur.rstrip())
            cur = "" if t.isspace() else t
    if cur.strip():
        out.append(cur.rstrip())
    return out


def text_block(dr, x, y, text, f, fill, width, gap=6):
    for line in wrap(dr, text, f, width):
        dr.text((x, y), line, font=f, fill=fill)
        y += f.size + gap
    return y


def footer(canvas, y):
    """Add the attribution band below the figure and crop the whole."""
    dr = ImageDraw.Draw(canvas)
    f = font(22)
    dr.line([(16, y + 8), (WID - 16, y + 8)], fill=(210, 209, 205), width=1)
    y = text_block(dr, 16, y + 18, CREDIT, f, SUB, WID - 32)
    return canvas.crop((0, 0, WID, y + 12))


def label(dr, xy, text, f, fill=(255, 255, 255), bg=(0, 0, 0)):
    """Label on top of a photo (a solid band, not translucent, as wide as the text)."""
    x, y = xy
    w = dr.textlength(text, font=f)
    dr.rectangle([x, y, x + w + 16, y + f.size + 14], fill=bg)
    dr.text((x + 8, y + 5), text, font=f, fill=fill)


def draw_traj(img, spec, lines):
    W, H = img.size
    d = ImageDraw.Draw(img)
    for pts, col, w in lines:
        g = np.array(pts, dtype=float).copy()
        g[:, 2] -= ORIGIN_H
        pp = [p for p in project(g, spec, W, H) if p is not None]
        if len(pp) > 1:
            d.line(pp, fill=col, width=w, joint="curve")


def bev_panel(size, gt, groups, min_range, f, rng=None, gt_ring=False, ego=(9, 6)):
    """Top view (forward is up, left is left). groups = [(list of xyz, color, width)]. The scale fits all points and the ego
    vehicle; the recorded drive (green) is drawn on top. Adds a scale bar. Pass rng = (xmin, xmax, ymin, ymax) to use that range."""
    Wp, Hp = size
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    if rng is None:
        rng = bev_range(gt, groups, min_range)
    xmin, xmax, ymin, ymax = rng
    top, bot, side = 50, 60, 24
    sc = min((Hp - top - bot) / (xmax - xmin), (Wp - 2 * side) / (ymax - ymin))
    # forward (x) is up, left (+y) is left. The center of the range goes to the center of the panel
    cx = Wp / 2 + (ymax + ymin) / 2 * sc
    cy = top + (Hp - top - bot) / 2 + (xmax + xmin) / 2 * sc
    bev = lambda p: [(cx - y * sc, cy - x * sc) for x, y in np.asarray(p)[:, :2]]  # noqa: E731
    for ps, col, w in groups:
        for p in ps:
            d.line(bev(p), fill=col, width=w, joint="curve")
    d.line(bev(gt), fill=GT, width=5, joint="curve")
    ex, ey = cx, cy  # ego vehicle (triangle, origin)
    if gt_ring:  # when the recorded drive barely moves: a green ring at its position 6.4 s later (around the triangle, not hidden by it)
        gx, gy = bev(gt)[-1]
        d.ellipse([gx - 15, gy - 15, gx + 15, gy + 15], outline=GT, width=4)
    a, b = ego  # size of the ego triangle (figure 6: (12, 8) as in the article; figure 11: (9, 6) to fit in the ring)
    d.polygon([(ex, ey - a), (ex - b, ey + b), (ex + b, ey + b)], outline=INK, fill=None, width=2)
    for L in (50, 20, 10, 5, 2, 1, 0.5):
        if L * sc <= Wp * 0.4:
            break
    x0, y0 = 16, Hp - 24
    d.line([(x0, y0), (x0 + L * sc, y0)], fill=INK, width=3)
    d.text((x0 + L * sc + 8, y0 - f.size + 4), f"{L:g} m", font=f, fill=INK)
    d.rectangle([0, 0, Wp - 1, Hp - 1], outline=(200, 200, 200))
    return im


def bev_range(gt, groups, min_range):
    allp = np.concatenate([np.zeros((1, 2)), np.asarray(gt)[:, :2]] + [np.asarray(p)[:, :2] for ps, _, _ in groups for p in ps])
    xmin, xmax = float(allp[:, 0].min()), float(allp[:, 0].max())
    ymin, ymax = float(allp[:, 1].min()), float(allp[:, 1].max())
    span = max(xmax - xmin, ymax - ymin, min_range)
    pad = 0.06 * span
    # widen a direction that is too narrow (both directions at least 40% of span)
    def widen(a, b):
        c, h = (a + b) / 2, max((b - a) / 2, 0.2 * span)
        return c - h - pad, c + h + pad
    return (*widen(xmin, xmax), *widen(ymin, ymax))


def ade(a, b):
    return sum(math.dist(p[:2], r[:2]) for p, r in zip(a, b)) / len(a)


# ---------------------------------------------------------------- figure 6
def fig6(out):
    log8, t0 = "9e9bcfb7", 8.5
    scene, runs = None, {"wide_ft": [], "nat": []}
    for line in open(N / "raw/cfg.jsonl"):
        r = json.loads(line)
        if not r.get("log_id", "").startswith(log8) or r.get("t0_rel_s") != t0:
            continue
        if r["kind"] == "scene":
            scene = r
        elif r["kind"] == "run" and r["config"] in runs and r["nav"] == "gt":
            runs[r["config"]].append(r)
    gt = np.array(scene["gt"])
    f_t, f_l, f_s, f_e = font(32, True), font(27, True), font(26), font(25)
    canvas = Image.new("RGB", (WID, 2400), BG)
    dr = ImageDraw.Draw(canvas)
    y = 14
    dr.text((16, y), f"Left turn (Pittsburgh, log {log8}, t0 = {t0} s, navigation text \"{scene['nav_info']['text']}\")", font=f_t, fill=INK)
    y += 56
    w, h = (WID - 48) // 2, int((WID - 48) // 2 * 9 / 16)
    heads = {"wide_ft": "Re-rendered input (front 120°, the adopted layout)",
             "nat": "AV2 cameras as is (front approx. 47°)"}
    for k, c in enumerate(("wide_ft", "nat")):
        smp = torch.load(N / f"raw/samples/{log8}_{t0}_{c}.pt")
        cams, imgs = smp["camera_indices"].tolist(), smp["image_frames"]
        last = {cc: i for i, cc in enumerate(cams)}
        front = Image.fromarray(imgs[last[1], 0].permute(1, 2, 0).numpy()).resize((w, h), Image.LANCZOS)
        draw_traj(front, B.CONFIGS[c][1], [(r["xyz"], COL[c], 3) for r in runs[c]] + [(gt, GT, 5)])
        x = 16 + k * (w + 16)
        dr.text((x, y), heads[c], font=f_l, fill=COL[c])
        canvas.paste(front, (x, y + 40))
    y += 40 + h + 24
    # lower row: top view + legend and reasoning
    pw, ph = 620, 520
    panel = bev_panel((pw, ph), gt, [([r["xyz"] for r in runs[c]], COL[c], 3) for c in ("nat", "wide_ft")], 20.0, f_e, ego=(12, 8))
    canvas.paste(panel, (16, y))
    dr.text((24, y + 10), "Top view (up = forward)", font=f_e, fill=SUB)
    tx, tw = 16 + pw + 32, WID - (16 + pw + 32) - 16
    yy = y
    m = {c: np.mean([r["ade"] for r in runs[c]]) for c in runs}
    dr.text((tx, yy), "■ Recorded drive", font=f_l, fill=GT); yy += 42
    dr.text((tx, yy), f"■ Prediction, re-rendered input (5 seeds): mean error {m['wide_ft']:.2f} m", font=f_l, fill=COL["wide_ft"]); yy += 42
    dr.text((tx, yy), f"■ Prediction, AV2 cameras as is (5 seeds): mean error {m['nat']:.2f} m", font=f_l, fill=COL["nat"]); yy += 60
    yy = text_block(dr, tx, yy, "Reasoning output by the model (seed 43; the same text with either input):", f_s, INK, tw)
    s43 = {c: next(r for r in runs[c] if r["seed"] == 43) for c in runs}
    assert s43["wide_ft"]["cot"] == s43["nat"]["cot"]
    yy = text_block(dr, tx + 16, yy + 4, s43["wide_ft"]["cot"], f_e, INK, tw - 16)
    yy = text_block(dr, tx, yy + 14, ("For the other seeds as well, the reasoning with either input says \"turn left at the intersection on a green light\"; "
                      "only the wording differs. The reasoning is the same, but the trajectory changes greatly with the field of view of the input."), f_s, SUB, tw)
    y = max(y + ph, yy) + 10
    footer(canvas, y).save(out)


# ---------------------------------------------------------------- figure 7
def fig7(out):
    logdir = next((N / "data/av2/val").glob("02678d04*"))
    t0rel = 5.0
    log = B.AV2Log(str(logdir))
    t0 = log.t0_from_rel(t0rel)
    f_t, f_l, f_s = font(32, True), font(27, True), font(25)
    rows = [("nat", "① AV2 cameras as is"),
            ("wide_pin", "② Adjacent cameras stitched into 120° (pinhole)"),
            ("wide_ft", "③ Adjacent cameras stitched into 120° (equidistant) … adopted")]
    slot = {0: "Left side", 1: "Front (wide)", 2: "Right side",
            3: "Left rear", 5: "Right rear", 6: "Front (tele)"}
    w = (WID - 16 * 4) // 3
    h = int(w * 9 / 16)
    canvas = Image.new("RGB", (WID, 3000), BG)
    dr = ImageDraw.Draw(canvas)
    y = 14
    dr.text((16, y), f"One scene (Pittsburgh, log 02678d04, t0 = {t0rel} s) as input built in 3 layouts", font=f_t, fill=INK)
    y += 50
    y = text_block(dr, 16, y, "Of the model's 6 input slots, the front 3 (left side, front, right side) are shown. Black areas are not covered by the AV2 cameras.", f_s, SUB, WID - 32)
    y += 10
    fl = font(24, True)
    for cfg, head in rows:
        dr.text((16, y), head, font=f_l, fill=COL["wide_ft"] if cfg == "wide_ft" else INK)
        y += 42
        for k, cam in enumerate((0, 1, 2)):
            img, _, _ = B.render(log, B.CONFIGS[cfg][cam], t0, DEV)
            im = Image.fromarray(img.permute(1, 2, 0).numpy()).resize((w, h), Image.LANCZOS)
            x = 16 + k * (w + 16)
            canvas.paste(im, (x, y))
            label(ImageDraw.Draw(canvas), (x, y + h - fl.size - 14), slot[cam], fl)
        if cfg == "wide_ft":
            dr.rectangle([12, y - 46, WID - 12, y + h + 4], outline=COL["wide_ft"], width=3)
        y += h + 24
    dr.text((16, y), "The other 3 slots of ③ (left rear, right rear, tele; tele is the central 30° cropped from the front camera)", font=f_l, fill=INK)
    y += 42
    for k, cam in enumerate((3, 5, 6)):
        img, _, _ = B.render(log, B.CONFIGS["wide_ft"][cam], t0, DEV)
        im = Image.fromarray(img.permute(1, 2, 0).numpy()).resize((w, h), Image.LANCZOS)
        x = 16 + k * (w + 16)
        canvas.paste(im, (x, y))
        label(ImageDraw.Draw(canvas), (x, y + h - fl.size - 14), slot[cam], fl)
    y += h + 10
    footer(canvas, y).save(out)


# ---------------------------------------------------------------- figure 11
def fig11(out):
    sid = "v_d46ff5df_8.5"
    rec = next(json.loads(l) for l in open(N / "raw/wide/wide.jsonl") if f'"sample_id": "{sid}"' in l)
    row = pd.read_csv(N / "raw/wide/scenes_t0.csv").set_index("scene_id").loc[sid]
    log = B.AV2Log(str(N / row.data_dir))
    spec = B.CONFIGS["wide_ft"][1]
    img, _, _ = B.render(log, spec, int(row.t0_ns), DEV)
    gt = np.array(rec["gt_xyz"])
    S = rec["sampled"]
    seeds = sorted(S["base"], key=int)
    f_t, f_l, f_s, f_e = font(32, True), font(26, True), font(26), font(24)
    canvas = Image.new("RGB", (WID, 3000), BG)
    dr = ImageDraw.Draw(canvas)
    y = 14
    dr.text((16, y), f"Pedestrian crossing (Detroit, log d46ff5df, t0 = 8.5 s, navigation text \"{rec['nav_text']}\")", font=f_t, fill=INK)
    y += 56
    fw = 1040
    fh = int(fw * 9 / 16)
    front = Image.fromarray(img.permute(1, 2, 0).numpy()).resize((fw, fh), Image.LANCZOS)
    vs = ("base", "fp8body", "fp8all")
    draw_traj(front, spec, [(S[v][s]["xyz"], COL[v], 3) for v in vs for s in seeds] + [(gt, GT, 5)])
    canvas.paste(front, (16, y))
    label(dr, (16, y), "Front slot (re-rendered input, t0)", font(24, True))
    pw = WID - 16 - fw - 16 - 16
    # top view: 3 columns, one per configuration (same scale), since overlaid lines would hide each other
    groups = [([S[v][s]["xyz"] for s in seeds], COL[v], 3) for v in vs]
    rng = bev_range(gt, groups, 3.0)
    head_h = 92
    colw = pw // 3
    px0 = 16 + fw + 16
    dr.rectangle([px0, y, px0 + pw - 1, y + fh - 1], fill="white", outline=(200, 200, 200))
    dr.text((px0 + 10, y + 8), "Top view (up = forward, 5 seeds)", font=f_e, fill=SUB)
    fl = font(22, True)
    rec_l = "◯ Recorded (stays stopped)"
    dr.text((px0 + 10, y + 40), rec_l, font=fl, fill=GT)
    dr.text((px0 + 10 + dr.textlength(rec_l, font=fl) + 24, y + 40), "△ Ego (t0)", font=fl, fill=INK)
    short = {"base": "BF16", "fp8body": "FP8\nbody only", "fp8all": "FP8 body +\noutput layer"}
    for k, v in enumerate(vs):
        sub = bev_panel((colw - 6, fh - head_h - 76), gt, [groups[k]], 3.0, font(20), rng=rng, gt_ring=True)
        canvas.paste(sub, (px0 + k * colw + 3, y + head_h + 72))
        dr.multiline_text((px0 + k * colw + colw // 2, y + head_h + 4), short[v], font=font(22, True), fill=TCOL[v],
                          anchor="ma", align="center", spacing=2)
    # only the upper part (front image and top views); the reasoning per seed is given in the article text
    y += fh + 6
    footer(canvas, y).save(out)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    which = sys.argv[2:] or ["fig6", "fig7", "fig11"]
    # file names of the published figures
    JPG = {"fig6": "fig06_left_turn", "fig7": "fig07_input_layouts", "fig11": "fig11_pedestrian_stop"}
    for w in which:
        p = out / f"{w}_src.png"
        globals()[w](p)
        j = out / (JPG[w] + ".jpg")
        Image.open(p).convert("RGB").save(j, quality=90, optimize=True, progressive=True, subsampling=0)
        print("saved", p, j, Image.open(p).size)

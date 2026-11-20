#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_fig_wide.py <wide.jsonl> <scenes_t0.csv> <scene_id> <variant> <out.png>
# Figure of a scene with a large difference in the wide scene comparison (§3.3).
#   Top left: front slot at t0 (wide_ft), overlaid with the recorded drive (green), BF16 (blue) and FP8 (orange/teal) trajectories for 5 seeds
#   Top right: top view. Bottom: reasoning per seed (BF16 and FP8; "same" if identical) and ADE vs. the recorded drive
#   Attribution: AV2 images (CC BY-NC-SA 4.0, © 2021 Argo AI, LLC)
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import av2_build as B  # noqa: E402
from av2_fig import ORIGIN_H, project  # noqa: E402

COL = {"gt": (0, 131, 0), "base": (42, 120, 214), "fp8body": (235, 104, 52), "fp8all": (27, 175, 122)}
NAME = {"base": "BF16", "fp8body": "FP8 body only", "fp8all": "FP8 body + output layer"}
FONT = "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf"


def font(size):
    for p in (FONT, "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


def ade(a, b):
    return sum(math.dist(p[:2], r[:2]) for p, r in zip(a, b)) / len(a)


def main():
    ev, scsv, sid, var, out = sys.argv[1:6]
    rec = next(json.loads(l) for l in open(ev) if f'"sample_id": "{sid}"' in l)
    row = pd.read_csv(scsv).set_index("scene_id").loc[sid]
    log = B.AV2Log(row.data_dir)
    spec = B.CONFIGS["wide_ft"][1]
    img, _, _ = B.render(log, spec, int(row.t0_ns))
    W, H = 960, 540
    front = Image.fromarray(img.permute(1, 2, 0).numpy()).resize((W, H), Image.BILINEAR)
    fd = ImageDraw.Draw(front)
    gt = np.array(rec["gt_xyz"])
    b, d = rec["sampled"]["base"], rec["sampled"][var]
    lines = [(np.array(b[s]["xyz"]), COL["base"], 2) for s in b] + [(np.array(d[s]["xyz"]), COL[var], 2) for s in d] + [(gt, COL["gt"], 5)]
    for pts, col, w in lines:
        g = pts.copy()
        g[:, 2] -= ORIGIN_H
        pp = [p for p in project(g, spec, W, H) if p is not None]
        if len(pp) > 1:
            fd.line(pp, fill=col, width=w)
    seeds = sorted(b, key=int)
    f15, f18 = font(15), font(18)
    canvas = Image.new("RGB", (W + 560, H + 70 + 22 * (2 * len(seeds) + 2)), (252, 252, 251))
    dr = ImageDraw.Draw(canvas)
    canvas.paste(front, (0, 0))
    dr.text((6, 4), "Front slot (t0)", fill="yellow", font=f18)
    # top view
    cx, cy = W + 280, 470
    allp = np.concatenate([gt[:, :2]] + [np.array(x["xyz"])[:, :2] for x in list(b.values()) + list(d.values())])
    sc = 250 / max(10.0, float(np.abs(allp).max()) * 1.1)
    bev = lambda p: [(cx - y * sc, cy - x * sc) for x, y in np.asarray(p)[:, :2]]  # noqa: E731
    dr.rectangle([W + 10, 0, W + 550, H], outline=(200, 200, 200))
    for s in seeds:
        dr.line(bev(b[s]["xyz"]), fill=COL["base"], width=2)
    for s in seeds:
        dr.line(bev(d[s]["xyz"]), fill=COL[var], width=2)
    dr.line(bev(gt), fill=COL["gt"], width=4)
    dr.text((W + 16, 6), "■ GT", fill=COL["gt"], font=f15)
    dr.text((W + 90, 6), f"■ {NAME['base']} (5 seeds)", fill=COL["base"], font=f15)
    dr.text((W + 260, 6), f"■ {NAME[var]} (5 seeds)", fill=COL[var], font=f15)
    dr.text((W + 16, H - 22), f"Top view ({250 / sc:.0f} m to edge)", fill=(82, 81, 78), font=f15)
    y = H + 8
    dr.text((6, y), f"{sid}  motion {rec['motion']}  {rec['city']}  nav text \"{rec['nav_text']}\"", fill=(11, 11, 11), font=f18)
    y += 28
    for s in seeds:
        dr.text((6, y), f"[BF16 s{s} ADE {ade(b[s]['xyz'], gt):.2f}] {b[s]['cot']}", fill=COL["base"], font=f15)
        y += 22
        same = d[s]["gen_ids"] == b[s]["gen_ids"]
        dr.text((6, y), f"[{NAME[var]} s{s} ADE {ade(d[s]['xyz'], gt):.2f}] {'(same reasoning as BF16)' if same else d[s]['cot']}",
                fill=COL[var], font=f15)
        y += 22
    dr.text((W + 560 - 6, y + 4), "Images: Argoverse 2 (CC BY-NC-SA 4.0, © 2021 Argo AI, LLC)", fill=(82, 81, 78), font=font(13), anchor="ra")
    canvas = canvas.crop((0, 0, canvas.width, y + 26))
    canvas.save(out)
    print("saved", out, canvas.size)


if __name__ == "__main__":
    main()

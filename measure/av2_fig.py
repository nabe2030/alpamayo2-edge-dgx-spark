#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_fig.py <eval.jsonl> <samples_dir> <log_id> <t0_rel_s> <out.png> <config>[,<config2>...] [nav]
# 1 場面の図: 左上 = 最初の組み立て方の前方スロット(t0)に、記録された走行(緑)と予測(seed ごと、赤系)を重ねたもの。
# 右上 = 上から見た図(履歴は灰、記録は緑、組み立て方ごとに色を変えた予測)。下 = ほかの 5 スロットと、各 seed の判断理由。
# 地面の高さ: AV2 の自車座標の原点は地面から約 0.32 m 上(地図の ground_height で確かめた)なので、軌跡の z から 0.32 m 引く。
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import av2_build as B  # noqa: E402

ORIGIN_H = 0.32
CAM_POS = np.array([1.63, 0.0, 1.40])  # 仮想の前方カメラの位置(ring_front_center の取り付け位置)
COLORS = [(220, 40, 40), (40, 90, 230), (230, 140, 0), (150, 0, 180), (0, 160, 160), (120, 120, 0), (200, 0, 120)]


def project(pts, spec, W, H):
    yaw = math.radians(spec["yaw"])
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    down = np.cross(fwd, right)
    d = np.asarray(pts) - CAM_POS
    xc, yc, zc = d @ right, d @ down, d @ fwd
    hf = math.radians(spec["hfov"])
    out = []
    for x, y, z in zip(xc, yc, zc):
        if z <= 0.5:
            out.append(None)
            continue
        if spec["model"] == "pinhole":
            f = (W / 2) / math.tan(hf / 2)
            out.append((W / 2 + f * x / z, H / 2 + f * y / z))
        else:
            f = (W / 2) / (hf / 2)
            th, ph = math.atan2(math.hypot(x, y), z), math.atan2(y, x)
            out.append((W / 2 + f * th * math.cos(ph), H / 2 + f * th * math.sin(ph)))
    return out


def main():
    ev, sdir, log_id, t0s, out = sys.argv[1:6]
    configs = sys.argv[6].split(",")
    nav = sys.argv[7] if len(sys.argv) > 7 else "gt"
    t0s = float(t0s)
    scene, runs = None, {c: [] for c in configs}
    for line in open(ev):
        r = json.loads(line)
        if r.get("log_id") != log_id or r.get("t0_rel_s") != t0s:
            continue
        if r["kind"] == "scene":
            scene = r
        elif r["kind"] == "run" and r["config"] in runs and r["nav"] == nav:
            runs[r["config"]].append(r)
    gt = np.array(scene["gt"])
    font = ImageFont.load_default(size=15)
    fontb = ImageFont.load_default(size=18)
    W, H = 960, 540
    canvas = Image.new("RGB", (W + 560, H + 190 + 18 * (2 + sum(len(v) for v in runs.values()))), "white")
    dr = ImageDraw.Draw(canvas)
    smp = torch.load(Path(sdir) / f"{log_id[:8]}_{t0s}_{configs[0]}.pt")
    cams, imgs = smp["camera_indices"].tolist(), smp["image_frames"]
    last = {c: i for i, c in enumerate(cams)}  # 各カメラの最後のコマ = t0
    spec = B.CONFIGS[configs[0]][1]
    front = Image.fromarray(imgs[last[1], 0].permute(1, 2, 0).numpy()).resize((W, H), Image.BILINEAR)
    fd = ImageDraw.Draw(front)
    for pts, col, wdt in [(gt, (0, 200, 0), 5)] + [(np.array(r["xyz"]), COLORS[0], 2) for r in runs[configs[0]]]:
        g = pts.copy()
        g[:, 2] -= ORIGIN_H
        pp = [p for p in project(g, spec, W, H) if p is not None]
        if len(pp) > 1:
            fd.line(pp, fill=col, width=wdt)
    canvas.paste(front, (0, 0))
    dr.text((6, 4), f"{configs[0]}: front slot at t0  (green = recorded, red = predicted)", fill="yellow", font=fontb)
    # 上から見た図
    S, cx, cy = 540, W + 280, 450
    rng = max(10.0, float(np.abs(np.concatenate([gt[:, :2]] + [np.array(r["xyz"])[:, :2] for v in runs.values() for r in v])).max()) * 1.1)
    sc = 250 / rng

    def bev(p):
        return [(cx - y * sc, cy - x * sc) for x, y in np.asarray(p)[:, :2]]

    dr.rectangle([W + 10, 0, W + 550, H], outline="gray")
    dr.line([(cx - 5, cy), (cx + 5, cy)], fill="black")
    dr.line(bev(gt), fill=(0, 180, 0), width=4)
    for k, c in enumerate(configs):
        for r in runs[c]:
            dr.line(bev(r["xyz"]), fill=COLORS[k], width=1)
        dr.text((W + 16, 6 + 18 * k), f"{c}: ADE mean {np.mean([r['ade'] for r in runs[c]]):.2f} m", fill=COLORS[k], font=font)
    dr.text((W + 16, H - 22), f"top view, {rng:.0f} m to edge; green = recorded", fill="black", font=font)
    # ほかのスロット
    tw, th = (W + 560) // 5, int((W + 560) // 5 * 9 / 16)
    names = {0: "0 front-left", 2: "2 front-right", 3: "3 rear-left", 5: "5 rear-right", 6: "6 tele"}
    for k, c in enumerate((0, 2, 3, 5, 6)):
        im = Image.fromarray(imgs[last[c], 0].permute(1, 2, 0).numpy()).resize((tw, th), Image.BILINEAR)
        canvas.paste(im, (k * tw, H + 4))
        dr.text((k * tw + 4, H + 6), names[c], fill="yellow", font=font)
    y = H + th + 10
    dr.text((6, y), f"log {log_id[:8]}  t0={t0s}s  motion={scene['motion']}  nav=\"{runs[configs[0]][0]['nav_text']}\"  "
                    f"const-velocity ADE {scene['cv_ade_fde'][0]:.2f} m", fill="black", font=fontb)
    y += 22
    for k, c in enumerate(configs):
        for r in runs[c]:
            dr.text((6, y), f"[{c} s{r['seed']} ADE {r['ade']:.2f}] {r['cot']}", fill=COLORS[k], font=font)
            y += 18
    canvas = canvas.crop((0, 0, canvas.width, y + 6))
    canvas.save(out)
    print("保存", out, canvas.size)


if __name__ == "__main__":
    main()

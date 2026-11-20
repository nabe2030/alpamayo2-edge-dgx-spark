#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_preview.py <log_dir> <t0_rel_s> <out_png> [config ...]
# Makes a check image with the t0 frames of the built slots (6 images) in one row per layout. Also shows time offsets.
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import av2_build as B  # noqa: E402

log = B.AV2Log(sys.argv[1])
t0 = log.t0_from_rel(float(sys.argv[2]))
configs = sys.argv[4:] or ["nat", "wide_pin", "wide_ft"]
TW, TH = 480, 270
font = ImageFont.load_default(size=16)
names = {0: "0 front-left(cross)", 1: "1 front(wide)", 2: "2 front-right(cross)", 3: "3 rear-left", 5: "5 rear-right", 6: "6 tele"}
canvas = Image.new("RGB", (TW * 6, (TH + 22) * len(configs)), "white")
for r, cfg in enumerate(configs):
    for k, cam in enumerate((0, 1, 2, 3, 5, 6)):
        img, t_used, stamps = B.render(log, B.CONFIGS[cfg][cam], t0)
        im = Image.fromarray(img.permute(1, 2, 0).numpy()).resize((TW, TH), Image.BILINEAR)
        canvas.paste(im, (k * TW, r * (TH + 22) + 22))
        lag = max((t0 - t) / 1e6 for t in stamps.values())
        ImageDraw.Draw(canvas).text((k * TW + 4, r * (TH + 22) + 2), f"{cfg} | {names[cam]} | lag<= {lag:.1f}ms", fill="black", font=font)
canvas.save(sys.argv[3])
print("saved", sys.argv[3], canvas.size)

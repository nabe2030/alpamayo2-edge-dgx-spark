#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# demo_frames_fp4.py <RT_tuned.jsonl> <RT_fp4.jsonl> <log_dir> <out_dir> [--fps 20] [--t-from 4.0] [--t-to 15.0] [--pass 1]
#                    [--device cuda] [--layout full|gif]
# demo_frames.py adapted for the FP4 video: the right side is "four optimizations + FP4 body (GPTQ), BF16 output layer"
# (records from demo_realtime.py --q fp4) instead of "FP8 body only". Output: docs/figs_av2/demo_realtime_fp4.gif. Changes from demo_frames.py: labels and color
# of the right side, a short sub-label next to each side's name in the GIF layout, one line in the bottom band on
# output quality (measured offline, see QUALITY_LINE), and a slightly smaller top view in the full layout to make room.
# Original description of demo_frames.py:
# Draws the frames (PNG) of the side-by-side video "official as is" vs. "tuned" (docs/figs_av2/demo_realtime.gif).
#   - The clock (tau) is real time (1x). Both sides show the same front image at tau (re-rendered front slot, wide_ft).
#   - The prediction shown is, from the records of the real-time loop of demo_realtime.py (pass --pass), the latest one
#     whose end time end_us is at or before tau, i.e. a prediction appears when its inference finishes. Its trajectory is in
#     the ego frame at the input time t0, so it is moved to the ego frame at tau with the recorded ego poses, and only the
#     points later than tau are drawn (an older prediction therefore looks shorter and shifted).
#   - Recorded drive (green): from tau to 6.4 s ahead (or to the end of the log).
#   - Colors: official BF16 = gray #8a8f98 and recorded drive = green as in article figure 11; FP4 body = orange #d9730d,
#     text in a darker orange #a65200 for contrast (the article charts use orange for 4-bit and blue for FP8; the FP8
#     video used blue #2f7fd1).
#   - The attribution (Argoverse 2, © 2021 Argo AI, LLC, CC BY-NC-SA 4.0, modified) and a note on the method are drawn at the bottom.
#   --layout gif: larger text so that it stays readable at 960 px wide, no top view, shorter bottom band.
# Making the video (ffmpeg, e.g. from `pip install imageio-ffmpeg`):
#   ffmpeg -framerate 20 -i out/f%04d.png -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p -movflags +faststart demo.mp4
#   ffmpeg -framerate 20 -i out_gif/f%04d.png -vf "fps=7,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" demo.gif
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import av2_build as B  # noqa: E402
from av2_fig import ORIGIN_H, project  # noqa: E402

W, H = 1920, 1080
BG = (252, 252, 251)
INK = (11, 11, 11)
SUB = (82, 81, 78)
GT = (0, 131, 0)
SIDE = {
    "tuned": dict(name="Official (tuned)", sub="BF16, upstream acceleration as is", col=(0x8a, 0x8f, 0x98), tcol=(0x5f, 0x64, 0x6c)),
    "fp4": dict(name="Four optimizations + FP4 body", sub="GPTQ 4-bit body, BF16 output layer", col=(0xd9, 0x73, 0x0d),
                tcol=(0xa6, 0x52, 0x00)),
}
SIDE["tuned"]["gsub"] = "BF16"
SIDE["fp4"]["gsub"] = "GPTQ"
# Output quality of the FP4 body, measured offline (not in this video): 400 AV2 scenes x 5 seeds against BF16 with the
# four optimizations, same seeds. Reasoning identical in 89.2 %; trajectory change 3.1 % of the seed-to-seed spread.
QUALITY_LINE = ("FP4 body vs. BF16, 400 scenes × 5 seeds (offline): same reasoning in 89%, "
                "trajectory change 3% of the seed-to-seed spread.")


def font(size, bold=False):
    p = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    return ImageFont.truetype(p, size, index=0)


def wrap(dr, text, f, width):
    out, cur = [], ""
    for w in text.split(" "):
        c = (cur + " " + w).strip()
        if dr.textlength(c, font=f) <= width or not cur:
            cur = c
        else:
            out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out


def load_rt(path, pas):
    recs = [json.loads(l) for l in open(path)]
    run = [r for r in recs if r.get("kind") == "run" and r.get("pass_") == pas]
    return sorted(run, key=lambda r: r["end_us"])


def to_world(log, t_ns, pts):
    xyz0, rot0 = log.poses(np.array([t_ns]))
    return rot0[0].apply(np.asarray(pts)) + xyz0[0]


def to_local(log, t_ns, world):
    xyz0, rot0 = log.poses(np.array([t_ns]))
    return rot0[0].inv().apply(np.asarray(world) - xyz0[0])


def draw_on_image(img, spec, lines):
    d = ImageDraw.Draw(img)
    Wi, Hi = img.size
    for pts, col, w in lines:
        if len(pts) < 2:
            continue
        g = np.array(pts, dtype=float).copy()
        g[:, 2] -= ORIGIN_H
        pp = [p for p in project(g, spec, Wi, Hi) if p is not None]
        if len(pp) > 1:
            d.line(pp, fill=col, width=w, joint="curve")


def bev(size, pred, gt, col, f):
    """Top view (forward is up). Fixed scale (40 m ahead, 30 m to the left, 12 m to the right; the ego vehicle sits right of center for this left turn)."""
    Wp, Hp = size
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    rng_f, rng_l, rng_r = 42.0, 30.0, 12.0
    sc = min((Hp - 30) / rng_f, Wp / (rng_l + rng_r))
    cx, cy = rng_l * sc, Hp - 18
    f2 = lambda p: [(cx - y * sc, cy - x * sc) for x, y in np.asarray(p)[:, :2]]  # noqa: E731
    for m in (10, 20, 30, 40):
        d.line([(0, cy - m * sc), (Wp, cy - m * sc)], fill=(232, 232, 230), width=1)
        d.text((Wp - 4, cy - m * sc - f.size - 2), f"{m} m", font=f, fill=(150, 150, 150), anchor="ra")
    if len(gt) > 1:
        d.line(f2(gt), fill=GT, width=5, joint="curve")
    if len(pred) > 1:
        d.line(f2(pred), fill=col, width=4, joint="curve")
    d.polygon([(cx, cy - 10), (cx - 7, cy + 7), (cx + 7, cy + 7)], outline=INK, width=2)
    d.rectangle([0, 0, Wp - 1, Hp - 1], outline=(200, 200, 200))
    return im


def gif_frame(tau, tau_ns, base, gt, R, spec, args, log, fm):
    """One frame for the GIF (drawn at 1920×1080 and scaled to 960 later; text stays at 15 px or more after scaling)."""
    canvas = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(canvas)
    f_title, f_head, f_big, f_txt, f_band = font(46, True), font(44, True), font(50, True), font(34), font(31)
    colw = (W - 48) // 2
    img_h = int(colw * 9 / 16)
    dr.text((20, 8), "Alpamayo 2 Edge on DGX Spark", font=f_title, fill=INK)
    dr.text((W - 20, 14), f"real time (1x)   t = {tau:4.1f} s", font=font(38), fill=SUB, anchor="ra")
    y0 = 72
    for j, k in enumerate(("tuned", "fp4")):
        sd = SIDE[k]
        x0 = 16 + j * (colw + 16)
        done = [r for r in R[k] if r["end_us"] <= tau * 1e6 + 1e-3]
        cur = done[-1] if done else None
        nup = len([r for r in done if r["end_us"] >= args.t_from * 1e6])
        dr.text((x0, y0), sd["name"], font=f_head, fill=sd["tcol"])
        dr.text((x0 + dr.textlength(sd["name"], font=f_head) + 16, y0 + 14), sd["gsub"], font=f_txt, fill=SUB)
        im = base.copy()
        pred = np.zeros((0, 3))
        if cur is not None:
            keep = cur["_pt_ns"] >= tau_ns
            pred = to_local(log, tau_ns, cur["_world"][keep]) if keep.any() else np.zeros((0, 3))
        draw_on_image(im, spec, [(gt, GT, 8), (pred, sd["col"], 8)])
        yi = y0 + 62
        canvas.paste(im, (x0, yi))
        fresh = cur is not None and tau * 1e6 - cur["end_us"] < 250_000
        dr.rectangle([x0 - 3, yi - 3, x0 + colw + 2, yi + img_h + 2], outline=sd["col"] if fresh else (210, 210, 205),
                     width=8 if fresh else 2)
        yb = yi + img_h + 12
        if cur is not None:
            ms = cur["e2e_ms"]
            dr.text((x0, yb), f"{ms:,.0f} ms per inference (≈{1000 / ms:.1f} Hz)", font=f_big, fill=sd["tcol"])
            age = tau - cur["t0_us"] / 1e6
            dr.text((x0, yb + 64), f"Shown: input from {age:.1f} s ago", font=f_txt, fill=INK)
            cot = cur["cot"] or ""
            if cot.strip() == "<NOMINAL>":
                cot = "<NOMINAL> (nominal driving, no reasoning text)"
            yy = yb + 110
            for line in wrap(dr, cot, f_txt, colw)[:2]:
                dr.text((x0, yy), line, font=f_txt, fill=SUB)
                yy += 42
            fm[k] = {"t0_us": cur["t0_us"], "end_us": cur["end_us"], "e2e_ms": round(ms, 1), "updates": nup}
    # bottom band (4 lines): legend and how predictions are shown / how they were measured / output quality /
    # source, license, modifications
    yl = H - 166
    dr.line([(16, yl - 8), (W - 16, yl - 8)], fill=(210, 209, 205), width=2)
    lx = 20
    for txt, col in (("■ recorded drive", GT), ("■ Official (tuned)", SIDE["tuned"]["col"]),
                     ("■ four optimizations + FP4 body", SIDE["fp4"]["col"])):
        dr.text((lx, yl), txt, font=f_band, fill=col)
        lx += dr.textlength(txt, font=f_band) + 24
    dr.text((lx, yl), "Prediction shown when its inference ends.", font=f_band, fill=SUB)
    dr.text((20, yl + 40), "Times and predictions measured on this log on DGX Spark (camera re-rendering not included).",
            font=f_band, fill=SUB)
    dr.text((20, yl + 80), QUALITY_LINE, font=f_band, fill=SUB)
    dr.text((20, yl + 120), "Images: Argoverse 2 Sensor Dataset (© 2021 Argo AI, LLC), CC BY-NC-SA 4.0; re-rendered, resized, annotated.",
            font=f_band, fill=SUB)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rt_tuned")
    ap.add_argument("rt_fp4")
    ap.add_argument("log_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--t-from", type=float, default=4.0)
    ap.add_argument("--t-to", type=float, default=15.0)
    ap.add_argument("--pass", dest="pas", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    # gif = layout for the GIF (larger text so that it stays readable at 960 px wide, no top view, shorter bottom band)
    ap.add_argument("--layout", choices=["full", "gif"], default="full")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = B.AV2Log(args.log_dir)
    spec = B.CONFIGS["wide_ft"][1]
    o = log.t_rel_origin_ns
    R = {"tuned": load_rt(args.rt_tuned, args.pas), "fp4": load_rt(args.rt_fp4, args.pas)}
    # input t0 (µs from the start of the log) -> time of the front-center frame (ns), matched as in AV2StreamSource
    fc = log.cams["ring_front_center"]["ts"]

    def t0_ns(t_us):
        target = o + int(t_us) * 1000
        return int(fc[int(np.argmin(np.abs(fc - target)))])

    # convert the predictions to world coordinates (time of point i = t0 + 0.1 s × (i + 1))
    for k, recs in R.items():
        for r in recs:
            tn = t0_ns(r["t0_us"])
            r["_t0_ns"] = tn
            r["_world"] = to_world(log, tn, r["xyz"])
            r["_pt_ns"] = tn + (np.arange(len(r["xyz"])) + 1) * 100_000_000
    f_title, f_head, f_sub, f_big, f_txt, f_small, f_tiny = (font(34, True), font(34, True), font(22), font(30, True),
                                                             font(24), font(20), font(18))
    colw = (W - 48) // 2
    img_h = int(colw * 9 / 16)
    n = int(round((args.t_to - args.t_from) * args.fps)) + 1
    meta = []
    for i in range(n):
        tau = args.t_from + i / args.fps
        tau_ns = o + int(round(tau * 1e9))
        frame_ns = log.frame_at_or_before("ring_front_center", tau_ns)
        cam, _, _ = B.render(log, spec, tau_ns, args.device)
        base = Image.fromarray(cam.permute(1, 2, 0).cpu().numpy()).resize((colw, img_h), Image.LANCZOS)
        # recorded drive (from tau to 6.4 s ahead, or to the end of the log)
        hi = min(tau_ns + 6_400_000_000, int(log.pose_ts[-1]))
        ts = np.arange(tau_ns, hi, 100_000_000)
        gw, _ = log.poses(ts)
        gt = to_local(log, tau_ns, gw)
        if args.layout == "gif":
            fm = {"tau": round(tau, 3), "frame_ns": frame_ns}
            canvas = gif_frame(tau, tau_ns, base, gt, R, spec, args, log, fm)
            canvas.save(out / f"f{i:04d}.png")
            meta.append(fm)
            continue
        canvas = Image.new("RGB", (W, H), BG)
        dr = ImageDraw.Draw(canvas)
        dr.text((24, 14), "Alpamayo 2 Edge on one DGX Spark (GB10): the same drive, two setups", font=f_title, fill=INK)
        dr.text((W - 24, 22), f"t = {tau:4.1f} s   real time (1x)", font=f_sub, fill=SUB, anchor="ra")
        y0 = 66
        fm = {"tau": round(tau, 3), "frame_ns": frame_ns}
        for j, k in enumerate(("tuned", "fp4")):
            sd = SIDE[k]
            x0 = 16 + j * (colw + 16)
            done = [r for r in R[k] if r["end_us"] <= tau * 1e6 + 1e-3]
            cur = done[-1] if done else None
            nup = len([r for r in done if r["end_us"] >= args.t_from * 1e6])
            dr.text((x0, y0), sd["name"], font=f_head, fill=sd["tcol"])
            dr.text((x0 + dr.textlength(sd["name"], font=f_head) + 14, y0 + 10), sd["sub"], font=f_sub, fill=SUB)
            im = base.copy()
            pred = np.zeros((0, 3))
            if cur is not None:
                keep = cur["_pt_ns"] >= tau_ns
                pred = to_local(log, tau_ns, cur["_world"][keep]) if keep.any() else np.zeros((0, 3))
            draw_on_image(im, spec, [(gt, GT, 6), (pred, sd["col"], 6)])
            yi = y0 + 50
            canvas.paste(im, (x0, yi))
            # highlight the frame for 0.25 s right after a new prediction appears
            fresh = cur is not None and tau * 1e6 - cur["end_us"] < 250_000
            dr.rectangle([x0 - 3, yi - 3, x0 + colw + 2, yi + img_h + 2], outline=sd["col"] if fresh else (210, 210, 205),
                         width=6 if fresh else 2)
            # below: top view + text
            yb = yi + img_h + 14
            bw = 280
            canvas.paste(bev((bw, 280), pred, gt, sd["col"], f_tiny), (x0, yb))
            tx = x0 + bw + 20
            tw = colw - bw - 20
            if cur is not None:
                ms = cur["e2e_ms"]
                dr.text((tx, yb), f"{ms:,.0f} ms per inference  (≈{1000 / ms:.1f} Hz)", font=f_big, fill=sd["tcol"])
                age = tau - cur["t0_us"] / 1e6
                dr.text((tx, yb + 44), f"Shown prediction: input at t = {cur['t0_us'] / 1e6:.1f} s ({age:.1f} s old)",
                        font=f_txt, fill=INK)
                dr.text((tx, yb + 76), f"Predictions finished since t = {args.t_from:.1f} s: {nup}", font=f_txt, fill=INK)
                yy = yb + 118
                dr.text((tx, yy), "Reasoning:", font=f_small, fill=SUB)
                yy += 28
                cot = cur["cot"] or ""
                if cot.strip() == "<NOMINAL>":  # model output as is, with a short explanation
                    cot = "<NOMINAL>  (model output for nominal driving: no reasoning text)"
                for line in wrap(dr, cot, f_txt, tw)[:4]:
                    dr.text((tx, yy), line, font=f_txt, fill=INK)
                    yy += 32
                fm[k] = {"t0_us": cur["t0_us"], "end_us": cur["end_us"], "e2e_ms": round(ms, 1), "updates": nup}
        # legend and notes (bottom band)
        yl = H - 118
        dr.line([(16, yl - 10), (W - 16, yl - 10)], fill=(210, 209, 205), width=1)
        lx = 24
        for txt, col in (("■ recorded drive", GT), ("■ prediction, Official (tuned)", SIDE["tuned"]["col"]),
                         ("■ prediction, four optimizations + FP4 body", SIDE["fp4"]["col"])):
            dr.text((lx, yl), txt, font=f_small, fill=col)
            lx += dr.textlength(txt, font=f_small) + 28
        dr.text((lx, yl), "Each prediction is shown from the moment its inference finished, moved to the current pose.",
                font=f_small, fill=SUB)
        dr.text((24, yl + 30), "Inference times and predictions were measured on this log on DGX Spark, running back to back "
                "(each inference starts on the latest input when the previous one ends; camera re-rendering not included).",
                font=f_tiny, fill=SUB)
        dr.text((24, yl + 56), QUALITY_LINE, font=f_tiny, fill=SUB)
        dr.text((24, yl + 82), "Images: Argoverse 2 Sensor Dataset (© 2021 Argo AI, LLC), CC BY-NC-SA 4.0 "
                "(creativecommons.org/licenses/by-nc-sa/4.0). Camera images re-rendered to a virtual camera, resized, "
                "trajectories and text overlaid.", font=f_tiny, fill=SUB)
        canvas.save(out / f"f{i:04d}.png")
        meta.append(fm)
    (out / "frames.json").write_text(json.dumps({"fps": args.fps, "t_from": args.t_from, "t_to": args.t_to,
                                                 "pass": args.pas, "layout": args.layout, "frames": meta}, indent=0))
    print("frames", n)


if __name__ == "__main__":
    main()

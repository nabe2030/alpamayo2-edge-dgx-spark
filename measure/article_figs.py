#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# article_figs.py <out_dir> [fig6|fig7|fig11 ...] — 記事に載せる AV2 の画像入りの図(図 6・7・11)を、これまでの記録から描き直す。
#   図 6 : 左折の場面(ログ 9e9bcfb7、t0 = 8.5 s)。描き直した入力(wide_ft)と、AV2 のカメラをそのまま(nat)の比較。
#          元の図 figs/fig_9e9bcfb7_wide_ft_vs_nat.png と同じ記録(raw/cfg.jsonl、raw/samples の入力)を使う。
#   図 7 : 同じ場面(ログ 02678d04、t0 = 5.0 s)を 3 通りの組み立て方で作った入力。元の図 figs/preview_02678d04.png と同じ描き方
#          (av2_build.render)。前の 3 スロットを 3 案で並べ、採用した wide_ft だけ残りの 3 スロットも見せる。
#   図 11: 歩行者が横断中の場面(v_d46ff5df_8.5)。元の図 wide_v_d46ff5df_8.5_fp8all.png と同じ記録(raw/wide/wide.jsonl)。
#          本体だけ FP8 の 5 seed も足した(記録にはもともとある)。
#   どの図も、図の中の説明は日本語(A2E_FIG_LANG=en で英語)、モデルの判断理由の英文はそのまま。図の下に帰属表示(Argoverse 2、© 2021 Argo AI, LLC、
#   CC BY-NC-SA 4.0、加工したこと)を入れる。記事の幅(800 px 前後)に縮めても読めるよう、幅 1600 px・本文 26 px 以上にした。
#
# 使い方(env.sh を読んだあと。PYTHONPATH に av2_input/ と measure/ が入っていること):
#   $PY article_figs.py <out_dir>                        # 日本語ラベル → <out_dir>/fig06_left_turn_ja.jpg など
#   A2E_FIG_LANG=en $PY article_figs.py <out_dir>        # 英語ラベル   → <out_dir>/fig06_left_turn.jpg など
#   どちらも <out_dir>/<図>_src.png(縮める前の PNG)も書く。JPEG は品質 90、4:4:4。
# 環境変数: A2E_WORK = 記録とデータの置き場所(既定はこのファイルの 1 つ上)。A2E_FIG_DEVICE = 描き直しの device(既定 cpu)。
# 読む記録(A2E_WORK の下):
#   図 6 : raw/cfg.jsonl(組み立て方の比較の記録。kind=scene の行に正解の軌跡、kind=run の行に各 seed の軌跡・ADE・判断理由)と、
#          raw/samples/<ログの先頭 8 文字>_<t0>_<組み立て方>.pt(そのときの入力。image_frames と camera_indices)
#   図 7 : data/av2/val/02678d04…(AV2 のログ。av2_fetch.py で取る)
#   図 11: raw/wide/wide.jsonl と raw/wide/scenes_t0.csv(measure/run_wide.sh の記録)と、そのログ
# フォント: Noto Sans CJK(/usr/share/fonts/opentype/noto/、Ubuntu なら fonts-noto-cjk)。
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
# 描き直しの device。既定は CPU(GPU のほかの仕事の邪魔をしない)。A2E_FIG_DEVICE=cuda で GPU
DEV = __import__("os").environ.get("A2E_FIG_DEVICE", "cpu")
# 図の文字の言語。既定は日本語。A2E_FIG_LANG=en で英語(英文記事の用語。図の中身は同じで、文字だけ変える)
EN = __import__("os").environ.get("A2E_FIG_LANG", "ja") == "en"


def T(ja, en):
    return en if EN else ja


WID = 1600
BG = (252, 252, 251)
INK = (11, 11, 11)
SUB = (82, 81, 78)
GT = (0, 131, 0)
COL = {"wide_ft": (214, 40, 40), "nat": (42, 100, 214),
       # 図 11 の 3 構成: 記事のグラフと同じ色。BF16 = #8a8f98、本体だけ = #2f7fd1、本体と出力層 = #2a9d8f
       "base": (0x8a, 0x8f, 0x98), "fp8body": (0x2f, 0x7f, 0xd1), "fp8all": (0x2a, 0x9d, 0x8f)}
# 文字の色(BF16 の灰色は文字には薄いので、濃いめの灰色にする)
TCOL = {**COL, "base": (0x5f, 0x64, 0x6c)}
NAME = {"base": "BF16", "fp8body": T("本体だけ FP8", "FP8 body only"), "fp8all": T("本体と出力層 FP8", "FP8 body + output layer")}
CREDIT = T("画像: Argoverse 2 Sensor Dataset(© 2021 Argo AI, LLC)、CC BY-NC-SA 4.0 "
           "(https://creativecommons.org/licenses/by-nc-sa/4.0/)。元のカメラ画像を加工した"
           "(仮想カメラへの描き直し、切り出し・縮小、軌跡と文字の重ね描き)。",
           "Images: Argoverse 2 Sensor Dataset (© 2021 Argo AI, LLC), CC BY-NC-SA 4.0 "
           "(https://creativecommons.org/licenses/by-nc-sa/4.0/). The original camera images were modified "
           "(re-rendered to virtual cameras, cropped and resized, trajectories and text overlaid).")


def font(size, bold=False):
    cands = (["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"] if bold else []) + [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf"]
    for p in cands:
        if Path(p).exists():
            return ImageFont.truetype(p, size, index=0)  # .ttc の 0 番 = JP
    raise SystemExit("日本語のフォントが無い")


def wrap(dr, text, f, width):
    """幅 width に収まるように折り返す(英文は語で、日本語は文字で)。"""
    import re
    out, cur = [], ""
    # 英数字の語はひとまとまり、空白は区切り、それ以外(日本語)は 1 文字ずつ
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
    """図の下に帰属の帯を足して、全体を切り詰める。"""
    dr = ImageDraw.Draw(canvas)
    f = font(22)
    dr.line([(16, y + 8), (WID - 16, y + 8)], fill=(210, 209, 205), width=1)
    y = text_block(dr, 16, y + 18, CREDIT, f, SUB, WID - 32)
    return canvas.crop((0, 0, WID, y + 12))


def label(dr, xy, text, f, fill=(255, 255, 255), bg=(0, 0, 0)):
    """写真の上に置くラベル(半透明ではなく塗りの帯。文字の幅ぶん全体に敷く)。"""
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
    """上から見た図(前が上、左が左)。groups = [(xyz のリスト, 色, 太さ)]。全部の点と自車が収まるように縮尺を決め、
    記録(緑)は一番上に描く。目盛りの棒を入れる。rng = (xmin, xmax, ymin, ymax) を渡すと、その範囲にそろえる。"""
    Wp, Hp = size
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    if rng is None:
        rng = bev_range(gt, groups, min_range)
    xmin, xmax, ymin, ymax = rng
    top, bot, side = 50, 60, 24
    sc = min((Hp - top - bot) / (xmax - xmin), (Wp - 2 * side) / (ymax - ymin))
    # 前(x)が上、左(+y)が左。範囲の真ん中を絵の真ん中に置く
    cx = Wp / 2 + (ymax + ymin) / 2 * sc
    cy = top + (Hp - top - bot) / 2 + (xmax + xmin) / 2 * sc
    bev = lambda p: [(cx - y * sc, cy - x * sc) for x, y in np.asarray(p)[:, :2]]  # noqa: E731
    for ps, col, w in groups:
        for p in ps:
            d.line(bev(p), fill=col, width=w, joint="curve")
    d.line(bev(gt), fill=GT, width=5, joint="curve")
    ex, ey = cx, cy  # 自車(三角、原点)
    if gt_ring:  # 記録がほとんど動かないとき: 6.4 秒後の記録の位置に緑の輪(三角を囲み、三角に隠れない)
        gx, gy = bev(gt)[-1]
        d.ellipse([gx - 15, gy - 15, gx + 15, gy + 15], outline=GT, width=4)
    a, b = ego  # 自車の三角の大きさ(図 6 は記事に載せた版と同じ (12, 8)、図 11 は輪の中に収める (9, 6))
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
    # 細すぎる向きは広げる(縦横どちらも span の 4 割以上)
    def widen(a, b):
        c, h = (a + b) / 2, max((b - a) / 2, 0.2 * span)
        return c - h - pad, c + h + pad
    return (*widen(xmin, xmax), *widen(ymin, ymax))


def ade(a, b):
    return sum(math.dist(p[:2], r[:2]) for p, r in zip(a, b)) / len(a)


# ---------------------------------------------------------------- 図 6
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
    dr.text((16, y), T(f"左折の場面(ピッツバーグ、ログ {log8}、t0 = {t0} 秒、ナビ文 \"{scene['nav_info']['text']}\")",
                        f"Left turn (Pittsburgh, log {log8}, t0 = {t0} s, navigation text \"{scene['nav_info']['text']}\")"), font=f_t, fill=INK)
    y += 56
    w, h = (WID - 48) // 2, int((WID - 48) // 2 * 9 / 16)
    heads = {"wide_ft": T("描き直した入力(前方 120°、採用した組み立て方)", "Re-rendered input (front 120°, the adopted layout)"),
             "nat": T("AV2 のカメラをそのまま(前方 約 47°)", "AV2 cameras as is (front approx. 47°)")}
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
    # 下段: 上から見た図 + 凡例と判断理由
    pw, ph = 620, 520
    panel = bev_panel((pw, ph), gt, [([r["xyz"] for r in runs[c]], COL[c], 3) for c in ("nat", "wide_ft")], 20.0, f_e, ego=(12, 8))
    canvas.paste(panel, (16, y))
    dr.text((24, y + 10), T("上から見た図(上が前方)", "Top view (up = forward)"), font=f_e, fill=SUB)
    tx, tw = 16 + pw + 32, WID - (16 + pw + 32) - 16
    yy = y
    m = {c: np.mean([r["ade"] for r in runs[c]]) for c in runs}
    dr.text((tx, yy), T("■ 記録された走行", "■ Recorded drive"), font=f_l, fill=GT); yy += 42
    dr.text((tx, yy), T(f"■ 描き直した入力での予測(5 seed)  記録との誤差 平均 {m['wide_ft']:.2f} m",
                                f"■ Prediction, re-rendered input (5 seeds): mean error {m['wide_ft']:.2f} m"), font=f_l, fill=COL["wide_ft"]); yy += 42
    dr.text((tx, yy), T(f"■ そのままの入力での予測(5 seed)  記録との誤差 平均 {m['nat']:.2f} m",
                                f"■ Prediction, AV2 cameras as is (5 seeds): mean error {m['nat']:.2f} m"), font=f_l, fill=COL["nat"]); yy += 60
    yy = text_block(dr, tx, yy, T("モデルが出した判断理由(seed 43。どちらの入力でも同じ文):", "Reasoning output by the model (seed 43; the same text with either input):"), f_s, INK, tw)
    s43 = {c: next(r for r in runs[c] if r["seed"] == 43) for c in runs}
    assert s43["wide_ft"]["cot"] == s43["nat"]["cot"]
    yy = text_block(dr, tx + 16, yy + 4, s43["wide_ft"]["cot"], f_e, INK, tw - 16)
    yy = text_block(dr, tx, yy + 14, T("ほかの seed も、どちらの入力でも「青信号で交差点を左折」の趣旨で、言い回しだけが違う。"
                      "判断は同じでも、軌跡は入力の画角で大きく変わる。",
                      "For the other seeds as well, the reasoning with either input says \"turn left at the intersection on a green light\"; "
                      "only the wording differs. The reasoning is the same, but the trajectory changes greatly with the field of view of the input."), f_s, SUB, tw)
    y = max(y + ph, yy) + 10
    footer(canvas, y).save(out)


# ---------------------------------------------------------------- 図 7
def fig7(out):
    logdir = next((N / "data/av2/val").glob("02678d04*"))
    t0rel = 5.0
    log = B.AV2Log(str(logdir))
    t0 = log.t0_from_rel(t0rel)
    f_t, f_l, f_s = font(32, True), font(27, True), font(25)
    rows = [("nat", T("① AV2 のカメラをそのまま入れる", "① AV2 cameras as is")),
            ("wide_pin", T("② 隣り合うカメラをつないで 120°(ピンホール)", "② Adjacent cameras stitched into 120° (pinhole)")),
            ("wide_ft", T("③ 隣り合うカメラをつないで 120°(等距離射影)… 採用", "③ Adjacent cameras stitched into 120° (equidistant) … adopted"))]
    slot = {0: T("左横", "Left side"), 1: T("前方(広角)", "Front (wide)"), 2: T("右横", "Right side"),
            3: T("左後ろ", "Left rear"), 5: T("右後ろ", "Right rear"), 6: T("前方(望遠)", "Front (tele)")}
    w = (WID - 16 * 4) // 3
    h = int(w * 9 / 16)
    canvas = Image.new("RGB", (WID, 3000), BG)
    dr = ImageDraw.Draw(canvas)
    y = 14
    dr.text((16, y), T(f"同じ場面(ピッツバーグ、ログ 02678d04、t0 = {t0rel} 秒)を 3 通りの組み立て方で作った入力",
                        f"One scene (Pittsburgh, log 02678d04, t0 = {t0rel} s) as input built in 3 layouts"), font=f_t, fill=INK)
    y += 50
    y = text_block(dr, 16, y, T("モデルの 6 つの入力スロットのうち、前の 3 つ(左横・前方・右横)を並べた。黒い所は、AV2 のカメラで覆えない部分。",
                               "Of the model's 6 input slots, the front 3 (left side, front, right side) are shown. Black areas are not covered by the AV2 cameras."), f_s, SUB, WID - 32)
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
    dr.text((16, y), T("③ の残りの 3 スロット(左後ろ・右後ろ・望遠。望遠は前方のカメラの中央 30° を切り出す)",
                        "The other 3 slots of ③ (left rear, right rear, tele; tele is the central 30° cropped from the front camera)"), font=f_l, fill=INK)
    y += 42
    for k, cam in enumerate((3, 5, 6)):
        img, _, _ = B.render(log, B.CONFIGS["wide_ft"][cam], t0, DEV)
        im = Image.fromarray(img.permute(1, 2, 0).numpy()).resize((w, h), Image.LANCZOS)
        x = 16 + k * (w + 16)
        canvas.paste(im, (x, y))
        label(ImageDraw.Draw(canvas), (x, y + h - fl.size - 14), slot[cam], fl)
    y += h + 10
    footer(canvas, y).save(out)


# ---------------------------------------------------------------- 図 11
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
    dr.text((16, y), T(f"歩行者が横断中の場面(デトロイト、ログ d46ff5df、t0 = 8.5 秒、ナビ文 \"{rec['nav_text']}\")",
                        f"Pedestrian crossing (Detroit, log d46ff5df, t0 = 8.5 s, navigation text \"{rec['nav_text']}\")"), font=f_t, fill=INK)
    y += 56
    fw = 1040
    fh = int(fw * 9 / 16)
    front = Image.fromarray(img.permute(1, 2, 0).numpy()).resize((fw, fh), Image.LANCZOS)
    vs = ("base", "fp8body", "fp8all")
    draw_traj(front, spec, [(S[v][s]["xyz"], COL[v], 3) for v in vs for s in seeds] + [(gt, GT, 5)])
    canvas.paste(front, (16, y))
    label(dr, (16, y), T("前方のスロット(描き直した入力、t0)", "Front slot (re-rendered input, t0)"), font(24, True))
    pw = WID - 16 - fw - 16 - 16
    # 上から見た図: 構成ごとに 3 列(同じ縮尺)。重ねると線が隠れるため
    groups = [([S[v][s]["xyz"] for s in seeds], COL[v], 3) for v in vs]
    rng = bev_range(gt, groups, 3.0)
    head_h = 92
    colw = pw // 3
    px0 = 16 + fw + 16
    dr.rectangle([px0, y, px0 + pw - 1, y + fh - 1], fill="white", outline=(200, 200, 200))
    dr.text((px0 + 10, y + 8), T("上から見た図(上が前方、5 seed)", "Top view (up = forward, 5 seeds)"), font=f_e, fill=SUB)
    fl = font(22, True)
    rec_l = T("◯ 記録(止まったまま)", "◯ Recorded (stays stopped)")
    dr.text((px0 + 10, y + 40), rec_l, font=fl, fill=GT)
    dr.text((px0 + 10 + dr.textlength(rec_l, font=fl) + 24, y + 40), T("△ 自車(t0)", "△ Ego (t0)"), font=fl, fill=INK)
    short = {"base": "BF16", "fp8body": T("本体だけ\nFP8", "FP8\nbody only"), "fp8all": T("本体と\n出力層 FP8", "FP8 body +\noutput layer")}
    for k, v in enumerate(vs):
        sub = bev_panel((colw - 6, fh - head_h - 76), gt, [groups[k]], 3.0, font(20), rng=rng, gt_ring=True)
        canvas.paste(sub, (px0 + k * colw + 3, y + head_h + 72))
        dr.multiline_text((px0 + k * colw + colw // 2, y + head_h + 4), short[v], font=font(22, True), fill=TCOL[v],
                          anchor="ma", align="center", spacing=2)
    y += fh + 22
    fx = [float(np.asarray(S[v][s]["xyz"])[-1, 0]) for v in vs for s in seeds]
    mn = lambda v: f"{v:+.1f}".replace("-", "−")  # noqa: E731  マイナスは「−」
    y = text_block(dr, 16, y, T(f"記録の車は止まったまま。予測の 6.4 秒後の前後の位置は、どの構成でも {mn(min(fx))}〜{mn(max(fx))} m"
                                "(わずかに前後する程度。後ろ向きの seed もある)。"
                                "判断理由(英文はモデルの出力のまま)は seed ごとに、上から BF16、本体だけ FP8、本体と出力層 FP8:",
                                f"The recorded vehicle stays stopped. The predicted longitudinal position 6.4 s later is {mn(min(fx))} to {mn(max(fx))} m "
                                "in every configuration (only slight movement back and forth; some seeds move backward). "
                                "Reasoning (as output by the model) per seed, from top: BF16, FP8 body only, FP8 body + output layer:"),
                   f_s, INK, WID - 32)
    y += 8
    for s in seeds:
        b = S["base"][s]
        dr.text((16, y), f"seed {s}", font=f_l, fill=INK)
        x0 = 140
        y = text_block(dr, x0, y, f"BF16: {b['cot']}", f_e, TCOL["base"], WID - x0 - 16, gap=4)
        for v in ("fp8body", "fp8all"):
            same = S[v][s]["gen_ids"] == b["gen_ids"]
            t = f"{NAME[v]}: " + (T("(BF16 と同じ判断理由)", "(same reasoning as BF16)") if same else S[v][s]["cot"])
            y = text_block(dr, x0, y, t, f_e, TCOL[v], WID - x0 - 16, gap=4)
        y += 10
    footer(canvas, y).save(out)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    which = sys.argv[2:] or ["fig6", "fig7", "fig11"]
    # 公開した図の名前(英語ラベルは元の名前、日本語ラベルは _ja を付ける)
    JPG = {"fig6": "fig06_left_turn", "fig7": "fig07_input_layouts", "fig11": "fig11_pedestrian_stop"}
    for w in which:
        p = out / f"{w}_src.png"
        globals()[w](p)
        j = out / (JPG[w] + ("" if EN else "_ja") + ".jpg")
        Image.open(p).convert("RGB").save(j, quality=90, optimize=True, progressive=True, subsampling=0)
        print("保存", p, j, Image.open(p).size)

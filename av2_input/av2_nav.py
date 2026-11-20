#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_nav.py — AV2 にはナビ文が無いので、記録された走行(t0 からログの終わりまで)からナビ文を作る。
#
# 書式は NVIDIA のデータセットの公開の説明(HF のデータセットカード「Navigation Labels」)に合わせる:
#   "<maneuver> in Nm"(N m 先の操作、250 m 以内)、"<maneuver>"(操作の地点にいる間)、"Continue straight"(250 m 以内に操作なし)
# 使う操作は公開の例にある Turn left / Turn right だけ。
#
# 作り方: t0 の向きからの向きの変化が 30 度を超えた所を「曲がった」とし、そこから戻って 10 度未満だった最後の点を
# 曲がり始めとする。曲がり始めまでの道のりが 5 m 以下なら距離なし、それより先なら "in Nm"(整数 m)。
# AV2 の地図(lane segment の is_intersection)で、曲がり始め〜曲がった所のどこかが交差点の中かも記録する。
# 弱点: ログが短い(t0 から先は長くて 14 s ほど)ので、その先の曲がり角は分からない(「Continue straight」に寄る)。
import json

import numpy as np
from scipy.spatial.transform import Rotation as R

TURN_DEG, ONSET_DEG, AT_M = 30.0, 10.0, 5.0


def _intersection_polys(log):
    """地図の交差点の中のレーンの多角形(city 座標の xy)。"""
    if hasattr(log, "_ipolys"):
        return log._ipolys
    path = next((log.root / "map").glob("log_map_archive_*.json"))
    m = json.loads(path.read_text())
    polys = []
    for ls in m["lane_segments"].values():
        if not ls.get("is_intersection"):
            continue
        left = [(p["x"], p["y"]) for p in ls["left_lane_boundary"]]
        right = [(p["x"], p["y"]) for p in ls["right_lane_boundary"]]
        polys.append(np.array(left + right[::-1]))
    log._ipolys = polys
    return polys


def _inside(pt, poly):
    x, y = pt
    xs, ys = poly[:, 0], poly[:, 1]
    xj, yj = np.roll(xs, 1), np.roll(ys, 1)
    cross = ((ys > y) != (yj > y)) & (x < (xj - xs) * (y - ys) / (yj - ys + 1e-12) + xs)
    return bool(cross.sum() % 2)


def nav_from_future(log, t0_ns: int) -> dict:
    t_end = int(log.pose_ts[-1])
    offs = np.arange(0, (t_end - t0_ns) // 1000 - 1, 100_000)
    xyz, rot = log.local_traj(t0_ns, offs)
    yaw = np.unwrap(R.from_matrix(rot).as_euler("zyx")[:, 0])
    dyaw = np.degrees(yaw - yaw[0])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1))])
    info = {"horizon_s": round(offs[-1] / 1e6, 1), "horizon_m": round(float(s[-1]), 1)}
    hit = np.nonzero(np.abs(dyaw) >= TURN_DEG)[0]
    if len(hit) == 0:
        info.update(maneuver=None, text="Continue straight")
        return info
    i_turn = int(hit[0])
    below = np.nonzero(np.abs(dyaw[: i_turn + 1]) < ONSET_DEG)[0]
    i_start = int(below[-1]) if len(below) else 0
    left = dyaw[i_turn] > 0
    dist = float(s[i_start])
    man = "Turn left" if left else "Turn right"
    text = man if dist <= AT_M else f"{man} in {int(round(dist))}m"
    # 交差点の中か(曲がり始め〜曲がった所の点を city 座標に戻して調べる)
    cxyz, _ = log.poses(t0_ns + offs[i_start: i_turn + 1] * 1000)
    polys = _intersection_polys(log)
    in_x = any(_inside(p[:2], poly) for p in cxyz[::3] for poly in polys
               if poly[:, 0].min() - 1 <= p[0] <= poly[:, 0].max() + 1 and poly[:, 1].min() - 1 <= p[1] <= poly[:, 1].max() + 1)
    info.update(maneuver=man, dist_m=round(dist, 1), text=text, in_intersection=in_x,
                turn_deg=round(float(dyaw[i_turn]), 1))
    return info


def variant(info: dict, nav: str) -> str:
    """ナビ文の作り方の候補。
    gt:       記録された走行から(上の作り方)
    map:      gt と同じだが、曲がる所が地図の交差点の外なら道なりのカーブとみなして "Continue straight"
    straight: いつも "Continue straight"(ナビの情報なし)
    flip:     左右を入れ替えた文(直進の場面では "Turn left in 30m")。ナビ文が効いているかの確かめ
    """
    if nav == "gt":
        return info["text"]
    if nav == "map":
        return info["text"] if info.get("maneuver") is None or info.get("in_intersection") else "Continue straight"
    if nav == "straight":
        return "Continue straight"
    if nav == "flip":
        t = info["text"]
        if "left" in t:
            return t.replace("left", "right")
        if "right" in t:
            return t.replace("right", "left")
        return "Turn left in 30m"
    raise ValueError(nav)

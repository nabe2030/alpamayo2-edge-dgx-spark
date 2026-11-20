#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_nav.py — AV2 has no navigation text, so build it from the recorded drive (from t0 to the end of the log).
#
# The format follows NVIDIA's public dataset description (the HF dataset card "Navigation Labels"):
#   "<maneuver> in Nm" (maneuver N m ahead, within 250 m), "<maneuver>" (while at the maneuver point), "Continue straight" (no maneuver within 250 m)
# Only the maneuvers in the public examples are used: Turn left / Turn right.
#
# Method: the point where the heading change from t0 exceeds 30 degrees is "turned"; going back from there, the last point under 10 degrees is
# the turn start. If the distance to the turn start is 5 m or less, no distance is given; beyond that, "in Nm" (integer m).
# Also records whether any point from turn start to "turned" lies inside an intersection on the AV2 map (is_intersection of lane segments).
# Weakness: logs are short (at most about 14 s after t0), so turns beyond that are unknown (biased toward "Continue straight").
import json

import numpy as np
from scipy.spatial.transform import Rotation as R

TURN_DEG, ONSET_DEG, AT_M = 30.0, 10.0, 5.0


def _intersection_polys(log):
    """Polygons of lanes inside map intersections (xy in city coordinates)."""
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
    # inside an intersection? (check points from turn start to "turned" converted back to city coordinates)
    cxyz, _ = log.poses(t0_ns + offs[i_start: i_turn + 1] * 1000)
    polys = _intersection_polys(log)
    in_x = any(_inside(p[:2], poly) for p in cxyz[::3] for poly in polys
               if poly[:, 0].min() - 1 <= p[0] <= poly[:, 0].max() + 1 and poly[:, 1].min() - 1 <= p[1] <= poly[:, 1].max() + 1)
    info.update(maneuver=man, dist_m=round(dist, 1), text=text, in_intersection=in_x,
                turn_deg=round(float(dyaw[i_turn]), 1))
    return info


def variant(info: dict, nav: str) -> str:
    """Candidate ways to build navigation text.
    gt:       from the recorded drive (method above)
    map:      same as gt, but if the turn is outside a map intersection it is treated as a curve in the road: "Continue straight"
    straight: always "Continue straight" (no navigation information)
    flip:     text with left and right swapped ("Turn left in 30m" in straight scenes). Checks whether the navigation text has an effect
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

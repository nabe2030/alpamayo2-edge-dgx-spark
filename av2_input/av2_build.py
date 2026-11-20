#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_build.py — builds Alpamayo 2 Edge input (a dict of the same form as the official loader's) from an Argoverse 2 Sensor log.
#
# Same conventions as the official loader (alpamayo2_edge/load_physical_aiavdataset.py):
#   - Camera IDs 0,1,2,3,5,6 (CAMERA_FRAME_COUNTS in sample.py). 1 (front) and 6 (front telephoto) get 5 frames at −2/−1/−0.5/−0.1/0 s,
#     the others 3 frames at −1/−0.1/0 s. Each frame is "the nearest frame at or before the requested time" (same direction as the official check).
#   - image_frames is a single uint8 tensor [22,1,3,H,W] (all images must have the same width and height).
#   - Ego history: 20 points from t0−1.9 to t0 in 0.1 s steps, into the t0 ego frame via the inverse of the t0 pose (xyz and 3x3 rotation).
#   - Times are integer microseconds. The model uses only the difference from the last frame (chat_template.relative_time_labels).
#
# AV2 cameras differ from the NVIDIA setup in orientation and field of view, and the front center is portrait (1550x2048), so every slot
# is re-rendered as a "virtual camera" (orientation yaw, horizontal field of view hfov, projection pinhole / ftheta, 1920x1080)
# from the 7 ring cameras using the calibration (intrinsics, distortion, mounting pose).
# When one virtual camera spans several source cameras, each pixel uses the source camera closest to its optical axis (seams are left as is).
# Areas no source camera covers are black.
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

RING = ("ring_front_center", "ring_front_left", "ring_front_right", "ring_side_left",
        "ring_side_right", "ring_rear_left", "ring_rear_right")
FRONT_OFFSETS_US = (-2_000_000, -1_000_000, -500_000, -100_000, 0)
OTHER_OFFSETS_US = (-1_000_000, -100_000, 0)
OUT_W, OUT_H = 1920, 1080  # 1080x1920, same as the official example (image_frames description in the README)

# Candidate slot layouts. Values are {camera ID: virtual camera spec}.
#   yaw: optical axis direction (degrees, left positive; ego frame x forward, y left), hfov: horizontal field of view (degrees), model: pinhole / ftheta
#   src: source cameras allowed (all ring cameras if omitted)
CONFIGS = {
    # no stitching: each AV2 camera, with only distortion corrected, cropped to 16:9
    "nat": {1: dict(yaw=0, hfov=47, model="pinhole", src=("ring_front_center",)),
            6: dict(yaw=0, hfov=30, model="pinhole", src=("ring_front_center",)),
            0: dict(yaw=45, hfov=62, model="pinhole", src=("ring_front_left",)),
            2: dict(yaw=-45, hfov=62, model="pinhole", src=("ring_front_right",)),
            3: dict(yaw=153, hfov=62, model="pinhole", src=("ring_rear_left",)),
            5: dict(yaw=-153, hfov=62, model="pinhole", src=("ring_rear_right",))},
    # stitched to match 120-degree and 70-degree fields of view (pinhole projection)
    "wide_pin": {1: dict(yaw=0, hfov=120, model="pinhole"),
                 6: dict(yaw=0, hfov=30, model="pinhole", src=("ring_front_center",)),
                 0: dict(yaw=90, hfov=120, model="pinhole"),
                 2: dict(yaw=-90, hfov=120, model="pinhole"),
                 3: dict(yaw=150, hfov=70, model="pinhole"),
                 5: dict(yaw=-150, hfov=70, model="pinhole")},
    # likewise, but wide uses equidistant projection (f-theta, r = f·θ). Telephoto is pinhole
    "wide_ft": {1: dict(yaw=0, hfov=120, model="ftheta"),
                6: dict(yaw=0, hfov=30, model="pinhole", src=("ring_front_center",)),
                0: dict(yaw=90, hfov=120, model="ftheta"),
                2: dict(yaw=-90, hfov=120, model="ftheta"),
                3: dict(yaw=150, hfov=70, model="ftheta"),
                5: dict(yaw=-150, hfov=70, model="ftheta")},
}


def derive_config(base: str, **slot_overrides) -> dict:
    """Build a layout that replaces only some slots of CONFIGS[base] (e.g. make telephoto the same as front)."""
    cfg = {k: dict(v) for k, v in CONFIGS[base].items()}
    for cam, spec in slot_overrides.items():
        cfg[int(cam.lstrip("c"))] = dict(spec)
    return cfg


CONFIGS["nat_notele"] = derive_config("nat", c6=CONFIGS["nat"][1])          # same image as front in the telephoto slot
CONFIGS["nat_side"] = derive_config("nat", c0=dict(yaw=99, hfov=62, model="pinhole", src=("ring_side_left",)),
                                    c2=dict(yaw=-99, hfov=62, model="pinhole", src=("ring_side_right",)))
CONFIGS["wide_ft_notele"] = derive_config("wide_ft", c6=CONFIGS["wide_ft"][1])
# -2: comparison of how to fill uncovered areas (fill: black = leave 0 / gray = 128 / edge = stretch the nearest visible pixel)
CONFIGS["wide_ft_gray"] = {k: dict(v, fill="gray") for k, v in CONFIGS["wide_ft"].items()}
CONFIGS["wide_ft_edge"] = {k: dict(v, fill="edge") for k, v in CONFIGS["wide_ft"].items()}
CONFIGS["wide_ft_cross60"] = derive_config("wide_ft", c0=dict(yaw=60, hfov=120, model="ftheta"),
                                           c2=dict(yaw=-60, hfov=120, model="ftheta"))


class AV2Log:
    """One log (a directory fetched with av2_fetch.py)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.log_id = self.root.name
        intr = pd.read_feather(self.root / "calibration/intrinsics.feather").set_index("sensor_name")
        ext = pd.read_feather(self.root / "calibration/egovehicle_SE3_sensor.feather").set_index("sensor_name")
        self.cams = {}
        for c in RING:
            i, e = intr.loc[c], ext.loc[c]
            ts = np.array(sorted(int(p.stem) for p in (self.root / "sensors/cameras" / c).glob("*.jpg")), dtype=np.int64)
            self.cams[c] = dict(fx=float(i.fx_px), fy=float(i.fy_px), cx=float(i.cx_px), cy=float(i.cy_px),
                                k=(float(i.k1), float(i.k2), float(i.k3)), W=int(i.width_px), H=int(i.height_px),
                                R=R.from_quat([e.qx, e.qy, e.qz, e.qw]).as_matrix(),  # camera → ego
                                t=np.array([e.tx_m, e.ty_m, e.tz_m]), ts=ts)
        p = pd.read_feather(self.root / "city_SE3_egovehicle.feather").sort_values("timestamp_ns")
        # overlaps less than 1 µs apart (per-sensor records at the same instant) keep only the first one
        keep = np.concatenate([[True], np.diff(p.timestamp_ns.values) >= 1000])
        p = p[keep]
        self.pose_ts = p.timestamp_ns.values.astype(np.int64)
        self._pose_t0 = int(self.pose_ts[0])  # converting to float64 loses ns digits, so keep offsets from the first time
        self.pose_xyz = p[["tx_m", "ty_m", "tz_m"]].values.astype(np.float64)
        self.pose_rot = R.from_quat(p[["qx", "qy", "qz", "qw"]].values)
        self._slerp = Slerp((self.pose_ts - self._pose_t0).astype(np.float64), self.pose_rot)
        # time origin (to make absolute_timestamps and t0_us relative times in microseconds): the first pose time.
        # The pose file is complete even in logs where only the needed frames were fetched, so this is the same regardless of how it was fetched (the model uses only differences).
        self.t_origin_ns = int(self.pose_ts[0])
        # the origin of "seconds from log start" in t0_from_rel is, as before, the first front-center frame (used with fully fetched logs)
        self.t_rel_origin_ns = int(self.cams["ring_front_center"]["ts"][0])

    # ---- time ----
    def t0_from_rel(self, t0_rel_s: float) -> int:
        """Set t0 to the time (ns) of the front-center frame closest to the given seconds from log start."""
        ts = self.cams["ring_front_center"]["ts"]
        target = self.t_rel_origin_ns + int(round(t0_rel_s * 1e9))
        # keep within the range where every camera has a frame 2 s earlier and poses extend 6.4 s ahead (scene selection counts from the first pose)
        lo = max(c["ts"][0] for c in self.cams.values()) + 2_000_000_000
        hi = int(self.pose_ts[-1]) - 6_400_000_000
        ok = ts[(ts >= lo) & (ts <= hi)]
        return int(ok[np.argmin(np.abs(ok - target))])

    def frame_at_or_before(self, cam: str, t_ns: int, tol_ns: int = 1_000_000) -> int:
        """The nearest frame at or before the requested time. AV2 frame times jitter by ±a few ns around 50 ms steps, so
        up to 1 ms counts as "the same time" (prevents jitter from dropping to the previous frame = 50 ms earlier)."""
        ts = self.cams[cam]["ts"]
        i = np.searchsorted(ts, t_ns + tol_ns, side="right") - 1
        if i < 0:
            raise ValueError(f"{cam}: no frame before {t_ns}")
        return int(ts[i])

    # ---- pose ----
    def poses(self, t_ns: np.ndarray):
        t_ns = np.asarray(t_ns, dtype=np.int64)
        if t_ns.min() < self.pose_ts[0] or t_ns.max() > self.pose_ts[-1]:
            raise ValueError("outside the pose range")
        rel = (t_ns - self._pose_t0).astype(np.float64)
        prel = (self.pose_ts - self._pose_t0).astype(np.float64)
        xyz = np.stack([np.interp(rel, prel, self.pose_xyz[:, j]) for j in range(3)], -1)
        return xyz, self._slerp(rel)

    def local_traj(self, t0_ns: int, offsets_us: np.ndarray):
        """Position and rotation in the t0 ego frame (same transform as the official loader: inverse of the t0 pose)."""
        xyz, rot = self.poses(t0_ns + np.asarray(offsets_us, dtype=np.int64) * 1000)
        xyz0, rot0 = self.poses(np.array([t0_ns]))
        inv = rot0[0].inv()
        return inv.apply(xyz - xyz0[0]), (inv * rot).as_matrix()

    # ---- images ----
    def image(self, cam: str, t_ns: int) -> np.ndarray:
        return np.asarray(Image.open(self.root / "sensors/cameras" / cam / f"{t_ns}.jpg").convert("RGB"))


def _undistorted_rmax(k, rd_max):
    """Find the undistorted radius (normalized coordinates) at the image corner, within the range where the distortion polynomial is monotonic."""
    r = np.linspace(0, 3, 30001)
    rd = r * (1 + k[0] * r**2 + k[1] * r**4 + k[2] * r**6)
    mono = np.argmax(np.diff(rd) <= 0) if np.any(np.diff(rd) <= 0) else len(r) - 1
    r, rd = r[: mono + 1], rd[: mono + 1]
    hit = np.nonzero(rd >= rd_max)[0]
    return float(r[hit[0]] if len(hit) else r[-1])


def ray_dirs(spec: dict, W: int = OUT_W, H: int = OUT_H) -> np.ndarray:
    """Ray direction for each virtual camera pixel (ego frame, [H,W,3]). Camera frame is x right, y down, z forward."""
    hf = math.radians(spec["hfov"])
    u = np.arange(W) + 0.5 - W / 2
    v = np.arange(H) + 0.5 - H / 2
    uu, vv = np.meshgrid(u, v)
    if spec["model"] == "pinhole":
        f = (W / 2) / math.tan(hf / 2)
        d = np.stack([uu / f, vv / f, np.ones_like(uu)], -1)
    elif spec["model"] == "ftheta":
        f = (W / 2) / (hf / 2)  # equidistant projection: pixel distance from center = f·θ
        th = np.hypot(uu, vv) / f
        ph = np.arctan2(vv, uu)
        d = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
    else:
        raise ValueError(spec["model"])
    yaw, pitch = math.radians(spec["yaw"]), math.radians(spec.get("pitch", 0.0))
    fwd = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch)])
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    down = np.cross(fwd, right)
    return d[..., :1] * right + d[..., 1:2] * down + d[..., 2:3] * fwd


def build_maps(log: AV2Log, spec: dict):
    """Which pixel of which source camera each virtual camera pixel reads. Returns: [(cam, grid[H,W,2] (−1 to 1), mask[H,W])]"""
    d = ray_dirs(spec)
    best = np.full(d.shape[:2], np.inf)
    owner = np.full(d.shape[:2], -1)
    uvs = []
    srcs = spec.get("src", RING)
    for j, c in enumerate(srcs):
        cm = log.cams[c]
        dc = d @ cm["R"]  # ego → camera (R is camera → ego, so multiply by its transpose from the right)
        z = dc[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            xn, yn = dc[..., 0] / z, dc[..., 1] / z
        r2 = xn**2 + yn**2
        rd_corner = math.hypot(max(cm["cx"], cm["W"] - cm["cx"]) / cm["fx"], max(cm["cy"], cm["H"] - cm["cy"]) / cm["fy"])
        rmax = _undistorted_rmax(cm["k"], rd_corner)
        k1, k2, k3 = cm["k"]
        fac = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        uu = cm["fx"] * xn * fac + cm["cx"]
        vv = cm["fy"] * yn * fac + cm["cy"]
        ok = (z > 1e-3) & (r2 < rmax**2) & (uu >= 0) & (uu <= cm["W"] - 1) & (vv >= 0) & (vv <= cm["H"] - 1)
        better = ok & (r2 < best)
        best[better] = r2[better]
        owner[better] = j
        uvs.append((uu, vv))
    out = []
    for j, c in enumerate(srcs):
        m = owner == j
        if not m.any():
            continue
        cm = log.cams[c]
        uu, vv = uvs[j]
        grid = np.stack([uu / (cm["W"] - 1) * 2 - 1, vv / (cm["H"] - 1) * 2 - 1], -1)
        grid[~m] = -2  # out of range (unused)
        out.append((c, torch.from_numpy(grid).float(), torch.from_numpy(m)))
    return out


_MAP_CACHE: dict = {}


def maps_for(log: AV2Log, spec: dict):
    geo = {k: v for k, v in spec.items() if k != "fill"}  # the fill method does not affect the mapping
    key = (log.log_id, json.dumps(geo, sort_keys=True))
    if key not in _MAP_CACHE:
        _MAP_CACHE[key] = build_maps(log, geo)
    return _MAP_CACHE[key]


_FILL_CACHE: dict = {}


def fill_index(log: AV2Log, spec: dict):
    """For each uncovered pixel, the position (flat index) of the nearest covered pixel. Used by the edge fill."""
    from scipy.ndimage import distance_transform_edt

    geo = {k: v for k, v in spec.items() if k != "fill"}
    key = (log.log_id, json.dumps(geo, sort_keys=True))
    if key not in _FILL_CACHE:
        covered = np.zeros((OUT_H, OUT_W), dtype=bool)
        for _, _, m in maps_for(log, geo):
            covered |= m.numpy()
        _, (iy, ix) = distance_transform_edt(~covered, return_indices=True)
        _FILL_CACHE[key] = (torch.from_numpy(~covered), torch.from_numpy(iy * OUT_W + ix).long())
    return _FILL_CACHE[key]


def render(log: AV2Log, spec: dict, t_req_ns: int, device: str = "cuda"):
    """One virtual camera image (uint8 [3,H,W]) and its time (the frame time of the main source camera = the camera covering the image center)."""
    maps = maps_for(log, spec)
    out = torch.zeros(3, OUT_H, OUT_W, device=device)
    center_cam, stamps = None, {}
    for c, grid, m in maps:
        t = log.frame_at_or_before(c, t_req_ns)
        stamps[c] = t
        img = torch.from_numpy(log.image(c, t)).to(device).permute(2, 0, 1)[None].float()
        s = F.grid_sample(img, grid[None].to(device), mode="bilinear", padding_mode="zeros", align_corners=True)[0]
        out = torch.where(m.to(device)[None], s, out)
        if m[OUT_H // 2, OUT_W // 2]:
            center_cam = c
    center_cam = center_cam or maps[0][0]
    fill = spec.get("fill", "black")
    if fill != "black":
        hole, idx = fill_index(log, spec)
        hole = hole.to(device)
        if fill == "gray":
            out = torch.where(hole[None], torch.full_like(out, 128.0), out)
        elif fill == "edge":
            flat = out.reshape(3, -1)
            out = flat[:, idx.to(device).reshape(-1)].reshape(3, OUT_H, OUT_W)
        else:
            raise ValueError(fill)
    return out.round().clamp(0, 255).to(torch.uint8).cpu(), stamps[center_cam], stamps


def build_sample(log: AV2Log, t0_ns: int, config: str | dict, nav_text: str, device: str = "cuda",
                 render_fn=None) -> dict:
    """Build an input dict of the same form as the official loader (passes validate_sample). Extra info goes in 'av2'."""
    from alpamayo2_edge.sample import CAMERA_PIXEL_BOUNDS, validate_sample

    cfg = CONFIGS[config] if isinstance(config, str) else config
    imgs, cams, stamps, detail = [], [], [], []
    for cam_id in (0, 1, 2, 3, 5, 6):  # same order as _CAMERAS in the official loader
        offs = FRONT_OFFSETS_US if cam_id in (1, 6) else OTHER_OFFSETS_US
        for off in offs:
            req = t0_ns + off * 1000
            img, t_used, src_stamps = (render_fn or render)(log, cfg[cam_id], req, device)
            imgs.append(img[None])
            cams.append(cam_id)
            stamps.append((t_used - log.t_origin_ns) // 1000)  # microseconds
            detail.append({"cam": cam_id, "req_us": (req - log.t_origin_ns) // 1000,
                           "lag_ms": {c: round((req - t) / 1e6, 1) for c, t in src_stamps.items()}})
    hxyz, hrot = log.local_traj(t0_ns, np.arange(-1_900_000, 1, 100_000))
    data = {
        "image_frames": torch.stack(imgs).contiguous(),  # [22,1,3,H,W]
        "camera_indices": torch.tensor(cams, dtype=torch.int64),
        "absolute_timestamps": torch.tensor(stamps, dtype=torch.int64),
        "ego_history_xyz": torch.from_numpy(hxyz).float().unsqueeze(0),
        "ego_history_rot": torch.from_numpy(hrot).float().unsqueeze(0),
        "nav_text": [nav_text],
        "camera_pixel_bounds": dict(CAMERA_PIXEL_BOUNDS),
        "min_pixels": 163840,
        "max_pixels": 196608,
        "input_profile": "camera_subsampled",
        "history_num_steps": 20,
        "clip_id": log.log_id,
        "t0_us": (t0_ns - log.t_origin_ns) // 1000,
        "provenance": {"dataset": "Argoverse 2 Sensor (CC BY-NC-SA 4.0)", "log_id": log.log_id,
                       "t0_ns": t0_ns, "config": cfg},
        "av2": {"frames": detail},
    }
    validate_sample(data)
    return data


def gt_future(log: AV2Log, t0_ns: int) -> np.ndarray:
    """Recorded future trajectory 0.1 to 6.4 s (64 points, t0 ego frame). Not shown to the model."""
    xyz, _ = log.local_traj(t0_ns, np.arange(100_000, 6_400_001, 100_000))
    return xyz

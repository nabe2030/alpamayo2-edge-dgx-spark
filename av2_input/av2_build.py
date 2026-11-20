#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_build.py — Argoverse 2 Sensor のログから、Alpamayo 2 Edge の入力(公式ローダと同じ形の dict)を組み立てる。
#
# 公式ローダ(alpamayo2_edge/load_physical_aiavdataset.py)と同じ決まり:
#   - カメラ ID 0,1,2,3,5,6(sample.py の CAMERA_FRAME_COUNTS)。1(前方)と 6(前方望遠)は −2/−1/−0.5/−0.1/0 s の 5 枚、
#     ほかは −1/−0.1/0 s の 3 枚。各コマは「求めた時刻と同じか、それより前の最も近いコマ」(公式の検査と同じ向き)。
#   - image_frames は [22,1,3,H,W] の uint8 の 1 つのテンソル(全部同じ縦横でなければならない)。
#   - 自車の履歴は t0−1.9 〜 t0 の 0.1 s きざみ 20 点を、t0 の姿勢の逆変換で t0 の自車座標へ(xyz と 3x3 回転)。
#   - 時刻は整数マイクロ秒。モデルが使うのは最後のコマとの差だけ(chat_template.relative_time_labels)。
#
# AV2 のカメラは向きと画角が NVIDIA の構成と違い、前方中央は縦長(1550x2048)なので、どのスロットも
# 「仮想カメラ」(向き yaw、水平の画角 hfov、射影 pinhole / ftheta、1920x1080)として、
# キャリブレーション(内部パラメータ・歪み・取り付け姿勢)を使ってリングカメラ 7 台から描き直す。
# 1 つの仮想カメラが複数の元カメラにまたがるときは、画素ごとに光軸に最も近い元カメラを使う(つなぎ目はそのまま)。
# 元カメラで覆えない部分は黒。
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
OUT_W, OUT_H = 1920, 1080  # 公式の例と同じ 1080x1920(README の image_frames の説明)

# スロットの組み立て方の候補。値は {カメラID: 仮想カメラの指定}。
#   yaw: 光軸の向き(度、左が正。自車座標の x 前・y 左)、hfov: 水平の画角(度)、model: pinhole / ftheta
#   src: 使ってよい元カメラ(省略時は全リング)
CONFIGS = {
    # 貼り合わせない: AV2 のカメラ 1 台ずつを歪みだけ直して 16:9 に切る
    "nat": {1: dict(yaw=0, hfov=47, model="pinhole", src=("ring_front_center",)),
            6: dict(yaw=0, hfov=30, model="pinhole", src=("ring_front_center",)),
            0: dict(yaw=45, hfov=62, model="pinhole", src=("ring_front_left",)),
            2: dict(yaw=-45, hfov=62, model="pinhole", src=("ring_front_right",)),
            3: dict(yaw=153, hfov=62, model="pinhole", src=("ring_rear_left",)),
            5: dict(yaw=-153, hfov=62, model="pinhole", src=("ring_rear_right",))},
    # 120 度・70 度の画角に合わせて貼り合わせる(射影はピンホール)
    "wide_pin": {1: dict(yaw=0, hfov=120, model="pinhole"),
                 6: dict(yaw=0, hfov=30, model="pinhole", src=("ring_front_center",)),
                 0: dict(yaw=90, hfov=120, model="pinhole"),
                 2: dict(yaw=-90, hfov=120, model="pinhole"),
                 3: dict(yaw=150, hfov=70, model="pinhole"),
                 5: dict(yaw=-150, hfov=70, model="pinhole")},
    # 同じく、広角は等距離射影(f-theta、r = f·θ)。望遠はピンホール
    "wide_ft": {1: dict(yaw=0, hfov=120, model="ftheta"),
                6: dict(yaw=0, hfov=30, model="pinhole", src=("ring_front_center",)),
                0: dict(yaw=90, hfov=120, model="ftheta"),
                2: dict(yaw=-90, hfov=120, model="ftheta"),
                3: dict(yaw=150, hfov=70, model="ftheta"),
                5: dict(yaw=-150, hfov=70, model="ftheta")},
}


def derive_config(base: str, **slot_overrides) -> dict:
    """CONFIGS[base] の一部のスロットだけを差し替えた組み立て方を作る(例: 望遠を前方と同じにする)。"""
    cfg = {k: dict(v) for k, v in CONFIGS[base].items()}
    for cam, spec in slot_overrides.items():
        cfg[int(cam.lstrip("c"))] = dict(spec)
    return cfg


CONFIGS["nat_notele"] = derive_config("nat", c6=CONFIGS["nat"][1])          # 望遠のスロットに前方と同じ絵
CONFIGS["nat_side"] = derive_config("nat", c0=dict(yaw=99, hfov=62, model="pinhole", src=("ring_side_left",)),
                                    c2=dict(yaw=-99, hfov=62, model="pinhole", src=("ring_side_right",)))
CONFIGS["wide_ft_notele"] = derive_config("wide_ft", c6=CONFIGS["wide_ft"][1])
# -2: 覆えない部分の埋め方の比較(fill: black = 0 のまま / gray = 128 / edge = 最も近い写っている画素を引き伸ばす)
CONFIGS["wide_ft_gray"] = {k: dict(v, fill="gray") for k, v in CONFIGS["wide_ft"].items()}
CONFIGS["wide_ft_edge"] = {k: dict(v, fill="edge") for k, v in CONFIGS["wide_ft"].items()}
CONFIGS["wide_ft_cross60"] = derive_config("wide_ft", c0=dict(yaw=60, hfov=120, model="ftheta"),
                                           c2=dict(yaw=-60, hfov=120, model="ftheta"))


class AV2Log:
    """1 本のログ(av2_fetch.py で取ったディレクトリ)。"""

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
                                R=R.from_quat([e.qx, e.qy, e.qz, e.qw]).as_matrix(),  # カメラ → 自車
                                t=np.array([e.tx_m, e.ty_m, e.tz_m]), ts=ts)
        p = pd.read_feather(self.root / "city_SE3_egovehicle.feather").sort_values("timestamp_ns")
        # 1 µs 未満しか離れていない重なり(センサーごとの記録が同じ瞬間に並ぶ)は先の 1 つだけ残す
        keep = np.concatenate([[True], np.diff(p.timestamp_ns.values) >= 1000])
        p = p[keep]
        self.pose_ts = p.timestamp_ns.values.astype(np.int64)
        self._pose_t0 = int(self.pose_ts[0])  # float64 に直すと ns の桁が落ちるので、先頭からの差で持つ
        self.pose_xyz = p[["tx_m", "ty_m", "tz_m"]].values.astype(np.float64)
        self.pose_rot = R.from_quat(p[["qx", "qy", "qz", "qw"]].values)
        self._slerp = Slerp((self.pose_ts - self._pose_t0).astype(np.float64), self.pose_rot)
        # 時刻の原点(absolute_timestamps と t0_us をマイクロ秒の相対時刻にする): 姿勢の最初の時刻。
        # 姿勢のファイルは要るコマだけ取ったログでも丸ごとあるので、取り方によらず同じ値になる(モデルは差しか使わない)。
        self.t_origin_ns = int(self.pose_ts[0])
        # t0_from_rel の「ログ先頭からの秒」の原点は、 と同じ前方中央の最初のコマ(丸ごと取ったログで使う)
        self.t_rel_origin_ns = int(self.cams["ring_front_center"]["ts"][0])

    # ---- 時刻 ----
    def t0_from_rel(self, t0_rel_s: float) -> int:
        """ログ先頭からの秒に最も近い、前方中央のコマの時刻(ns)を t0 にする。"""
        ts = self.cams["ring_front_center"]["ts"]
        target = self.t_rel_origin_ns + int(round(t0_rel_s * 1e9))
        # どのカメラにも 2 s 前のコマがあり、姿勢が 6.4 s 先まである範囲に収める(場面選びは姿勢の先頭から数えているため)
        lo = max(c["ts"][0] for c in self.cams.values()) + 2_000_000_000
        hi = int(self.pose_ts[-1]) - 6_400_000_000
        ok = ts[(ts >= lo) & (ts <= hi)]
        return int(ok[np.argmin(np.abs(ok - target))])

    def frame_at_or_before(self, cam: str, t_ns: int, tol_ns: int = 1_000_000) -> int:
        """求めた時刻と同じかそれより前の、最も近いコマ。AV2 のコマの時刻は 50 ms きざみから ±数 ns 揺れるので、
        1 ms までは「同じ時刻」とみなす(揺れで 1 つ前のコマ = 50 ms 前に落ちるのを防ぐ)。"""
        ts = self.cams[cam]["ts"]
        i = np.searchsorted(ts, t_ns + tol_ns, side="right") - 1
        if i < 0:
            raise ValueError(f"{cam}: {t_ns} より前のコマが無い")
        return int(ts[i])

    # ---- 姿勢 ----
    def poses(self, t_ns: np.ndarray):
        t_ns = np.asarray(t_ns, dtype=np.int64)
        if t_ns.min() < self.pose_ts[0] or t_ns.max() > self.pose_ts[-1]:
            raise ValueError("姿勢の範囲の外")
        rel = (t_ns - self._pose_t0).astype(np.float64)
        prel = (self.pose_ts - self._pose_t0).astype(np.float64)
        xyz = np.stack([np.interp(rel, prel, self.pose_xyz[:, j]) for j in range(3)], -1)
        return xyz, self._slerp(rel)

    def local_traj(self, t0_ns: int, offsets_us: np.ndarray):
        """t0 の自車座標での位置と回転(公式ローダと同じ変換: t0 の姿勢の逆変換)。"""
        xyz, rot = self.poses(t0_ns + np.asarray(offsets_us, dtype=np.int64) * 1000)
        xyz0, rot0 = self.poses(np.array([t0_ns]))
        inv = rot0[0].inv()
        return inv.apply(xyz - xyz0[0]), (inv * rot).as_matrix()

    # ---- 画像 ----
    def image(self, cam: str, t_ns: int) -> np.ndarray:
        return np.asarray(Image.open(self.root / "sensors/cameras" / cam / f"{t_ns}.jpg").convert("RGB"))


def _undistorted_rmax(k, rd_max):
    """歪みの多項式が単調な範囲で、画像の隅に写る歪みのない半径(正規化座標)を求める。"""
    r = np.linspace(0, 3, 30001)
    rd = r * (1 + k[0] * r**2 + k[1] * r**4 + k[2] * r**6)
    mono = np.argmax(np.diff(rd) <= 0) if np.any(np.diff(rd) <= 0) else len(r) - 1
    r, rd = r[: mono + 1], rd[: mono + 1]
    hit = np.nonzero(rd >= rd_max)[0]
    return float(r[hit[0]] if len(hit) else r[-1])


def ray_dirs(spec: dict, W: int = OUT_W, H: int = OUT_H) -> np.ndarray:
    """仮想カメラの画素ごとの光線の向き(自車座標、[H,W,3])。カメラ座標は x 右・y 下・z 前。"""
    hf = math.radians(spec["hfov"])
    u = np.arange(W) + 0.5 - W / 2
    v = np.arange(H) + 0.5 - H / 2
    uu, vv = np.meshgrid(u, v)
    if spec["model"] == "pinhole":
        f = (W / 2) / math.tan(hf / 2)
        d = np.stack([uu / f, vv / f, np.ones_like(uu)], -1)
    elif spec["model"] == "ftheta":
        f = (W / 2) / (hf / 2)  # 等距離射影: 中心からの画素距離 = f·θ
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
    """仮想カメラの各画素が、どの元カメラのどの画素を読むか。返り値: [(cam, grid[H,W,2] (−1〜1), mask[H,W])]"""
    d = ray_dirs(spec)
    best = np.full(d.shape[:2], np.inf)
    owner = np.full(d.shape[:2], -1)
    uvs = []
    srcs = spec.get("src", RING)
    for j, c in enumerate(srcs):
        cm = log.cams[c]
        dc = d @ cm["R"]  # 自車 → カメラ(R はカメラ → 自車なので転置を右から掛ける)
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
        grid[~m] = -2  # 範囲外(使わない)
        out.append((c, torch.from_numpy(grid).float(), torch.from_numpy(m)))
    return out


_MAP_CACHE: dict = {}


def maps_for(log: AV2Log, spec: dict):
    geo = {k: v for k, v in spec.items() if k != "fill"}  # 埋め方は写し方に関係しない
    key = (log.log_id, json.dumps(geo, sort_keys=True))
    if key not in _MAP_CACHE:
        _MAP_CACHE[key] = build_maps(log, geo)
    return _MAP_CACHE[key]


_FILL_CACHE: dict = {}


def fill_index(log: AV2Log, spec: dict):
    """覆えない画素ごとの、最も近い覆っている画素の位置(flat な添字)。edge の埋め方で使う。"""
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
    """仮想カメラの 1 枚(uint8 [3,H,W])と、その時刻(主な元カメラ = 画面の中央を写すカメラのコマの時刻)。"""
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
    """公式ローダと同じ形の入力 dict を作る(validate_sample に通る)。付帯情報は 'av2' に入れる。"""
    from alpamayo2_edge.sample import CAMERA_PIXEL_BOUNDS, validate_sample

    cfg = CONFIGS[config] if isinstance(config, str) else config
    imgs, cams, stamps, detail = [], [], [], []
    for cam_id in (0, 1, 2, 3, 5, 6):  # 公式ローダの _CAMERAS と同じ順
        offs = FRONT_OFFSETS_US if cam_id in (1, 6) else OTHER_OFFSETS_US
        for off in offs:
            req = t0_ns + off * 1000
            img, t_used, src_stamps = (render_fn or render)(log, cfg[cam_id], req, device)
            imgs.append(img[None])
            cams.append(cam_id)
            stamps.append((t_used - log.t_origin_ns) // 1000)  # マイクロ秒
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
    """記録された未来の軌跡 0.1〜6.4 s(64 点、t0 の自車座標)。モデルには見せない。"""
    xyz, _ = log.local_traj(t0_ns, np.arange(100_000, 6_400_001, 100_000))
    return xyz

#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_render_fast.py — 仮想カメラへの描き直しを速くした版。
#
# av2_build.render と同じ計算(同じ写し方の表、同じ grid_sample・torch.where・丸め)で、出力の画素は一致する
# (decode="pil" / "pil_threads" のとき。確かめ方は av2_render_check.py)。速くしたところ:
#   1. 写し方の表(grid と mask)を GPU に置いたままにする(これまでは 1 枚ごとに 1920x1080x2 の float を転送していた)
#   2. 元カメラの 1 コマの JPEG は 1 回だけ読む(同じコマを複数のスロットが使う: 前方中央は広角と望遠、
#      横のカメラは前と横の 2 スロット)。読んだコマは GPU に置き、2.2 s より古いものから捨てる
#   3. 新しく要るコマの JPEG を、スレッドで並べて読む(PIL の読み込みは GIL を外す)
#   4. 公式の validate_sample は「連続した CPU の uint8」を求める(GPU に置いたまま渡すと止まる)。描いた絵は GPU に置き、
#      歩ごとに 22 枚を使い回しの pinned メモリ 1 つへまとめて写す(out_mode="pinned"、既定)。
#      最初の版(out_mode="cpu")は 1 枚ずつ .cpu で写して持っていたが、推論と交互に動かすと .cpu() の所で跳ねが出た。
#   5. 同じ絵になる要求(スロット × 元カメラごとのコマの時刻)は 1 回だけ描く(av2_stream と同じ考え)
# 速いが同じにならない選択肢(既定では使わない):
#   decode="pil_draft": JPEG を 1/2 の大きさで読む(PIL の draft。DCT の段階で縮める)。写し方の表は元の画素座標の
#   まま(grid は −1〜1 なので大きさによらない)。出力の差は av2_render_check.py で測る。
#   decode="nvjpeg": torchvision.io.decode_jpeg(device="cuda")。使えるかどうかは環境しだい。
from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import av2_build as B


class FastRenderer:
    def __init__(self, log: B.AV2Log, config: str | dict = "wide_ft", device: str = "cuda",
                 decode: str = "pil_threads", threads: int = 8, keep_s: float = 2.2, out_mode: str = "pinned"):
        self.log, self.device, self.decode = log, device, decode
        # out_mode: "cpu" = 描いた絵を 1 枚ずつ .cpu で写して持つ(最初の版)、
        #           "pinned" = 描いた絵は GPU に置き、歩ごとに 22 枚を使い回しの pinned メモリ 1 つへまとめて写す
        #           (歩ごとのメモリ確保をなくす。描き直し込みの測定で、.cpu の所で 100〜1000 ms の跳ねが出たため)
        self.out_mode = out_mode
        self._buf = None
        self.cfg = B.CONFIGS[config] if isinstance(config, str) else config
        self.keep_ns = int(keep_s * 1e9)
        self.pool = ThreadPoolExecutor(threads) if decode in ("pil_threads", "pil_draft") else None
        self.maps = {}  # cam_id → [(cam, grid[1,H,W,2] GPU, mask[1,H,W] GPU, center?)]
        for cid, spec in self.cfg.items():
            ms = B.maps_for(log, spec)
            self.maps[cid] = [(c, g[None].to(device), m.to(device)[None], bool(m[B.OUT_H // 2, B.OUT_W // 2]))
                              for c, g, m in ms]
            if spec.get("fill", "black") != "black":
                raise ValueError("FastRenderer は黒の埋め方(wide_ft)だけ")
        self.frames: dict = {}  # (cam, t_ns) → GPU uint8 [1,3,H,W](float にするのは使うとき。メモリを 1/4 にする)
        self.out: dict = {}  # (cam_id, stamps) → (GPU uint8 [3,H,W], t_center)
        self.stats = {"decoded": 0, "rendered": 0, "decode_ms": 0.0, "warp_ms": 0.0}

    # ---- JPEG ----
    def _path(self, cam, t):
        return self.log.root / "sensors/cameras" / cam / f"{t}.jpg"

    def _read(self, key):
        cam, t = key
        if self.decode == "nvjpeg":
            from torchvision.io import decode_jpeg, read_file
            return key, read_file(str(self._path(cam, t)))
        im = Image.open(self._path(cam, t))
        if self.decode == "pil_draft":
            im.draft("RGB", (im.width // 2, im.height // 2))
        return key, np.asarray(im.convert("RGB"))

    def _load(self, keys):
        keys = [k for k in dict.fromkeys(keys) if k not in self.frames]
        if not keys:
            return
        t = time.perf_counter()
        if self.decode == "nvjpeg":
            from torchvision.io import decode_jpeg
            raw = [self._read(k) for k in keys]
            imgs = decode_jpeg([r for _, r in raw], device=self.device)
            for (k, _), im in zip(raw, imgs):
                self.frames[k] = im[None]
        else:
            it = self.pool.map(self._read, keys) if self.pool else map(self._read, keys)
            for k, a in it:
                self.frames[k] = torch.from_numpy(a).to(self.device).permute(2, 0, 1)[None]
        self.stats["decoded"] += len(keys)
        self.stats["decode_ms"] += (time.perf_counter() - t) * 1000

    def _evict(self, t_now):
        for k in [k for k in self.frames if k[1] < t_now - self.keep_ns]:
            del self.frames[k]
        for k in [k for k, v in self.out.items() if max(k[1]) < t_now - self.keep_ns]:
            del self.out[k]

    # ---- 描き直し ----
    def _warp(self, cid, stamps):
        out = torch.zeros(3, B.OUT_H, B.OUT_W, device=self.device)
        center = None
        for (c, grid, m, is_c), t in zip(self.maps[cid], stamps):
            s = F.grid_sample(self.frames[(c, t)].float(), grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
            out = torch.where(m, s, out)
            if is_c:
                center = t
        center = center if center is not None else stamps[0]
        u8 = out.round().clamp(0, 255).to(torch.uint8)
        return (u8.cpu() if self.out_mode == "cpu" else u8), center

    def frames_for(self, t0_ns):
        """t0 の 22 枚(公式ローダの順)。返り値: (CPU uint8 [22,1,3,H,W]、連続, カメラ ID の並び, 時刻(µs)の並び)"""
        reqs = []
        for cid in (0, 1, 2, 3, 5, 6):
            offs = B.FRONT_OFFSETS_US if cid in (1, 6) else B.OTHER_OFFSETS_US
            for off in offs:
                t_req = t0_ns + off * 1000
                stamps = tuple(self.log.frame_at_or_before(c, t_req) for c, _, _, _ in self.maps[cid])
                reqs.append((cid, stamps))
        todo = [r for r in dict.fromkeys(reqs) if r not in self.out]
        self._load([(c, t) for cid, st in todo for (c, _, _, _), t in zip(self.maps[cid], st)])
        t = time.perf_counter()
        for r in todo:
            self.out[r] = self._warp(*r)
        self.stats["rendered"] += len(todo)
        self.stats["warp_ms"] += (time.perf_counter() - t) * 1000  # 非同期なので目安(合計は呼び出し側で同期して測る)
        self._evict(t0_ns - 2_000_000_000)
        if self.out_mode == "cpu":
            imgs = torch.stack([self.out[r][0] for r in reqs])[:, None].contiguous()
        else:
            if self._buf is None:
                self._buf = torch.empty((len(reqs), 1, 3, B.OUT_H, B.OUT_W), dtype=torch.uint8, pin_memory=True)
            for i, r in enumerate(reqs):
                self._buf[i, 0].copy_(self.out[r][0], non_blocking=True)
            torch.cuda.synchronize()
            imgs = self._buf  # 次の歩で上書きする(呼び出し側は 1 歩ずつ使い終える)
        stamps_us = [(self.out[r][1] - self.log.t_origin_ns) // 1000 for r in reqs]
        return imgs, [r[0] for r in reqs], stamps_us

    def sample(self, t0_ns, nav_text):
        """build_sample と同じ形の dict。"""
        from alpamayo2_edge.sample import CAMERA_PIXEL_BOUNDS

        imgs, cams, stamps = self.frames_for(t0_ns)
        hxyz, hrot = self.log.local_traj(t0_ns, np.arange(-1_900_000, 1, 100_000))
        return {
            "image_frames": imgs,
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
            "clip_id": self.log.log_id,
            "t0_us": (t0_ns - self.log.t_origin_ns) // 1000,
        }


class AV2InlineSource:
    """連続の測定で、描き直しを推論と同じ流れの中で行う入力源(a2e_bench2 --av2-render inline)。
    sample(t0_us) を呼ぶたびに、その t0 で新しく要るコマだけを読み、描き直す(前の t0 の絵は使い回す)。
    ナビ文(記録された走行から作る)と t0 の対応は、測る区間の外で先に作っておく(経路の計画にあたる部分は測らない)。"""

    def __init__(self, log_dir, t0_list_us, config="wide_ft", nav="gt", device="cuda", decode="pil_threads",
                 out_mode="pinned"):
        import av2_nav as N

        self.log = B.AV2Log(log_dir)
        self.r = FastRenderer(self.log, config, device, decode=decode, out_mode=out_mode)
        fc = self.log.cams["ring_front_center"]["ts"]
        self.t0_ns, self.navs = {}, {}
        for t in t0_list_us:
            target = self.log.t_rel_origin_ns + int(t) * 1000
            j = int(np.argmin(np.abs(fc - target)))
            self.t0_ns[int(t)] = int(fc[j])
            self.navs[int(t)] = N.variant(N.nav_from_future(self.log, int(fc[j])), nav)
        self.config, self.nav, self.decode = config, nav, decode

    def reset(self):
        """パスの始めに、読んだコマと描いた絵を捨てる(パスごとに同じ条件で始める)。"""
        self.r.frames.clear()
        self.r.out.clear()

    def sample(self, t0_us):
        return self.r.sample(self.t0_ns[int(t0_us)], self.navs[int(t0_us)])

    def describe(self):
        return {"log_id": self.log.log_id, "config": self.config, "nav": self.nav, "render": "inline",
                "decode": self.decode, "out_mode": self.r.out_mode, "n_t0": len(self.t0_ns)}

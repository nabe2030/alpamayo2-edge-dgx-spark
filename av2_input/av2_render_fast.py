#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_render_fast.py — faster version of re-rendering to the virtual cameras.
#
# Same computation as av2_build.render (same mapping tables, same grid_sample, torch.where and rounding); output pixels match
# (with decode="pil" / "pil_threads"; checked with av2_render_check.py). What was made faster:
#   1. keep the mapping tables (grid and mask) on the GPU (previously a 1920x1080x2 float was transferred per image)
#   2. read the JPEG of each source camera frame only once (several slots use the same frame: front center for wide and telephoto,
#      side cameras for the front and side slots). Frames read are kept on the GPU and dropped once older than 2.2 s
#   3. read the newly needed JPEG frames in parallel with threads (PIL decoding releases the GIL)
#   4. the official validate_sample requires "contiguous CPU uint8" (passing GPU tensors stops it). Rendered images stay on the GPU,
#      and each step copies the 22 images together into one reused pinned buffer (out_mode="pinned", default).
#      The first version (out_mode="cpu") copied each image with .cpu and kept it, but interleaved with inference it caused spikes at .cpu().
#   5. requests that give the same image (slot × frame time per source camera) are rendered only once (same idea as av2_stream)
# Faster options that do not give identical output (not used by default):
#   decode="pil_draft": read JPEGs at 1/2 size (PIL draft; shrinks at the DCT stage). The mapping tables stay in the original pixel
#   coordinates (grid is −1 to 1, so independent of size). The output difference is measured with av2_render_check.py.
#   decode="nvjpeg": torchvision.io.decode_jpeg(device="cuda"). Availability depends on the environment.
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
        # out_mode: "cpu" = copy each rendered image with .cpu and keep it (first version),
        #           "pinned" = keep rendered images on the GPU and copy the 22 images per step together into one reused pinned buffer
        #           (removes per-step memory allocation; in measurements including re-rendering, spikes of 100 to 1000 ms appeared at .cpu)
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
                raise ValueError("FastRenderer supports only the black fill (wide_ft)")
        self.frames: dict = {}  # (cam, t_ns) → GPU uint8 [1,3,H,W] (converted to float only when used; 1/4 the memory)
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

    # ---- re-rendering ----
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
        """The 22 images at t0 (official loader order). Returns: (CPU uint8 [22,1,3,H,W], contiguous, camera ID list, time (µs) list)"""
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
        self.stats["warp_ms"] += (time.perf_counter() - t) * 1000  # asynchronous, so approximate (the caller synchronizes to measure the total)
        self._evict(t0_ns - 2_000_000_000)
        if self.out_mode == "cpu":
            imgs = torch.stack([self.out[r][0] for r in reqs])[:, None].contiguous()
        else:
            if self._buf is None:
                self._buf = torch.empty((len(reqs), 1, 3, B.OUT_H, B.OUT_W), dtype=torch.uint8, pin_memory=True)
            for i, r in enumerate(reqs):
                self._buf[i, 0].copy_(self.out[r][0], non_blocking=True)
            torch.cuda.synchronize()
            imgs = self._buf  # overwritten on the next step (the caller finishes using it one step at a time)
        stamps_us = [(self.out[r][1] - self.log.t_origin_ns) // 1000 for r in reqs]
        return imgs, [r[0] for r in reqs], stamps_us

    def sample(self, t0_ns, nav_text):
        """A dict of the same form as build_sample."""
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
    """Input source for continuous measurement that re-renders within the same flow as inference (a2e_bench2 --av2-render inline).
    Each call to sample(t0_us) reads and re-renders only the frames newly needed at that t0 (images from the previous t0 are reused).
    The mapping from t0 to navigation text (built from the recorded drive) is built beforehand, outside the measured interval (the route-planning part is not measured)."""

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
        """At the start of a pass, drop the frames read and images rendered (each pass starts under the same conditions)."""
        self.r.frames.clear()
        self.r.out.clear()

    def sample(self, t0_us):
        return self.r.sample(self.t0_ns[int(t0_us)], self.navs[int(t0_us)])

    def describe(self):
        return {"log_id": self.log.log_id, "config": self.config, "nav": self.nav, "render": "inline",
                "decode": self.decode, "out_mode": self.r.out_mode, "n_t0": len(self.t0_ns)}

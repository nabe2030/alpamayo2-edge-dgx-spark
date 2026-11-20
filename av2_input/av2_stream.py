#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_stream.py — input source that feeds AV2 input to the speed measurement (a2e_bench2.py).
#
#   AV2StreamSource(log_dir, t0_list_us, config="wide_ft", nav="gt")
#     .sample(t0_us) → a dict of the same form as the official loader's (passes validate_sample)
#
# Samples per t0 are assembled by table lookup. Re-rendering to the virtual cameras (JPEG loading + remapping on the GPU) is
# done in this preparation and is not counted in inference time. The re-rendering time is kept in self.render_ms.
#
# t0_list_us is microseconds from the log start (the first front-center frame). Aligned to front-center frame times (absorbing ±1 ms jitter),
# and limited to the range where every camera has a frame 2 s earlier. Speed measurement does not use the GT trajectory, so it can run to near the end of the log.
# Navigation text per t0 comes from av2_nav.nav_from_future (from the recorded drive; shorter lookahead near the end of the log).
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import av2_build as B  # noqa: E402
import av2_nav as N  # noqa: E402


class AV2StreamSource:
    def __init__(self, log_dir: str, t0_list_us: list[int], config: str = "wide_ft", nav: str = "gt",
                 device: str = "cuda"):
        self.log = B.AV2Log(log_dir)
        self.cfg = B.CONFIGS[config]
        self.config, self.nav, self.device = config, nav, device
        fc = self.log.cams["ring_front_center"]["ts"]
        lo = max(int(c["ts"][0]) for c in self.log.cams.values()) + 2_000_000_000
        self.t0_ns = {}
        for t in t0_list_us:
            target = self.log.t_rel_origin_ns + int(t) * 1000
            j = int(np.argmin(np.abs(fc - target)))
            if abs(int(fc[j]) - target) > 1_000_000 or fc[j] < lo:
                raise ValueError(f"t0={t} µs: no front-center frame, or 2 s of history unavailable")
            self.t0_ns[int(t)] = int(fc[j])
        self.pool: dict = {}
        self.render_ms: list = []  # re-rendering time per image (from JPEG loading through GPU remapping and copy back to CPU)
        t = time.perf_counter()
        for t0 in self.t0_ns.values():
            B.build_sample(self.log, t0, self.cfg, "x", device, render_fn=self._render)
        self.prepare_s = time.perf_counter() - t
        self.navs = {t: N.variant(N.nav_from_future(self.log, t0), nav) for t, t0 in self.t0_ns.items()}
        self.n_frames = len(self.pool)

    def _key(self, spec, t_req):
        """Render identical requests only once: key is (spec, frame time selected per source camera)."""
        maps = B.maps_for(self.log, spec)
        stamps = tuple(self.log.frame_at_or_before(c, t_req) for c, _, _ in maps)
        return json.dumps(spec, sort_keys=True), stamps

    def _render(self, log, spec, t_req, device):
        k = self._key(spec, t_req)
        if k not in self.pool:
            torch.cuda.synchronize()
            t = time.perf_counter()
            self.pool[k] = B.render(log, spec, t_req, device)
            torch.cuda.synchronize()
            self.render_ms.append((time.perf_counter() - t) * 1000)
        return self.pool[k]

    def sample(self, t0_us: int) -> dict:
        d = B.build_sample(self.log, self.t0_ns[int(t0_us)], self.cfg, self.navs[int(t0_us)], self.device,
                           render_fn=self._render)
        d.pop("av2", None)
        return d

    def describe(self) -> dict:
        r = np.array(self.render_ms)
        return {"log_id": self.log.log_id, "config": self.config, "nav": self.nav, "n_t0": len(self.t0_ns),
                "n_rendered": len(r), "prepare_s": round(self.prepare_s, 2),
                "render_ms_median": round(float(np.median(r)), 2) if len(r) else None,
                "render_ms_p90": round(float(np.percentile(r, 90)), 2) if len(r) else None}

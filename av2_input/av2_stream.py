#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_stream.py — 速さの測定(a2e_bench2.py)に AV2 の入力を渡すための入力源。
#
#   AV2StreamSource(log_dir, t0_list_us, config="wide_ft", nav="gt")
#     .sample(t0_us) → 公式ローダと同じ形の dict(validate_sample に通る)
#
# t0 ごとのサンプルは表引きで組み立てる。仮想カメラへの描き直し(JPEG の読み込み + GPU での写し替え)は
# この準備の中で行い、推論の時間には入れない。描き直しにかかった時間は self.render_ms に残す。
#
# t0_list_us はログの先頭(前方中央の最初のコマ)からのマイクロ秒。前方中央のコマの時刻に合わせ(±1 ms の揺れを吸収)、
# どのカメラにも 2 s 前のコマがある範囲に限る。速さの測定は正解の軌跡を使わないので、ログの終わり近くまで使える。
# ナビ文は t0 ごとに av2_nav.nav_from_future(記録された走行から。ログの終わり近くでは先が短い)。
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
                raise ValueError(f"t0={t} µs に前方中央のコマが無いか、2 s の履歴が取れない")
            self.t0_ns[int(t)] = int(fc[j])
        self.pool: dict = {}
        self.render_ms: list = []  # 1 枚ごとの描き直しの時間(JPEG の読み込みから GPU の写し替え、CPU へ戻すまで)
        t = time.perf_counter()
        for t0 in self.t0_ns.values():
            B.build_sample(self.log, t0, self.cfg, "x", device, render_fn=self._render)
        self.prepare_s = time.perf_counter() - t
        self.navs = {t: N.variant(N.nav_from_future(self.log, t0), nav) for t, t0 in self.t0_ns.items()}
        self.n_frames = len(self.pool)

    def _key(self, spec, t_req):
        """同じ絵になる要求は 1 回だけ描く: 鍵は (仕様, 元カメラごとに選ばれるコマの時刻)。"""
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

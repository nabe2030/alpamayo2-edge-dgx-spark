#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_render_check.py <out.json> <log_dir> [<log_dir> ...]
# Whether the faster re-rendering (av2_render_fast) produces the same images as the previous re-rendering (av2_build.render), and timing.
#   1. whether nvJPEG (the GPU version of torchvision.io.decode_jpeg) works in this environment
#   2. match: per log, 8 t0 points, build the 22 images both ways and compare pixels (decode = pil_threads / pil_draft / nvjpeg)
#   3. timing (with no inference running on the GPU): rendering all 22 images (from frame loading) / only the increment in continuous mode (10Hz, 1Hz)
#      compared with the previous method (av2_build.render one image at a time; like av2_stream, identical images are reused)
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import av2_build as B  # noqa: E402
import av2_render_fast as RF  # noqa: E402


def synced(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    r = fn()
    torch.cuda.synchronize()
    return r, (time.perf_counter() - t) * 1000


class OldIncremental:
    """The previous method (same as av2_stream.AV2StreamSource._render: identical images are reused; JPEGs are read per slot)"""

    def __init__(self, log):
        self.log, self.pool, self.cfg = log, {}, B.CONFIGS["wide_ft"]

    def _render(self, log, spec, t_req, device):
        maps = B.maps_for(log, spec)
        k = (json.dumps(spec, sort_keys=True), tuple(log.frame_at_or_before(c, t_req) for c, _, _ in maps))
        if k not in self.pool:
            self.pool[k] = B.render(log, spec, t_req, device)
        return self.pool[k]

    def sample(self, t0):
        return B.build_sample(self.log, t0, self.cfg, "x", render_fn=self._render)


def main():
    out = sys.argv[1]
    res = {"nvjpeg": None, "logs": {}}
    torch.zeros(1, device="cuda")
    # 1. nvJPEG
    try:
        from torchvision.io import decode_jpeg, read_file
        log0 = B.AV2Log(sys.argv[2])
        c = "ring_front_center"
        f = log0.root / "sensors/cameras" / c / f"{int(log0.cams[c]['ts'][100])}.jpg"
        g = decode_jpeg(read_file(str(f)), device="cuda")
        torch.cuda.synchronize()
        res["nvjpeg"] = {"ok": True, "shape": list(g.shape)}
    except Exception as e:  # noqa: BLE001
        res["nvjpeg"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    print("nvjpeg", res["nvjpeg"], flush=True)
    modes = ["pil_threads", "pil_draft", "pil"] + (["nvjpeg"] if res["nvjpeg"]["ok"] else [])
    for ld in sys.argv[2:]:
        log = B.AV2Log(ld)
        for cid in (0, 1, 2, 3, 5, 6):
            B.maps_for(log, B.CONFIGS["wide_ft"][cid])
        fc = log.cams["ring_front_center"]["ts"]
        lo = max(int(c["ts"][0]) for c in log.cams.values()) + 2_000_000_000
        ok = fc[fc >= lo]
        r = {}
        # 2. match (8 points)
        pts = [int(x) for x in ok[:: max(1, len(ok) // 8)][:8]]
        for mode in modes:
            diffs, same = [], 0
            for t0 in pts:
                ref = B.build_sample(log, t0, "wide_ft", "x")
                fr = RF.FastRenderer(log, "wide_ft", decode=mode)
                d = fr.sample(t0, "x")
                a, b = ref["image_frames"].int(), d["image_frames"].cpu().int()
                same += int(torch.equal(a, b) and torch.equal(ref["absolute_timestamps"], d["absolute_timestamps"])
                            and torch.equal(ref["camera_indices"], d["camera_indices"]))
                diffs.append(((a - b).abs().float().mean().item(), (a - b).abs().max().item()))
            r[f"match_{mode}"] = {"t0": len(pts), "identical": same,
                                  "mean_abs_max": round(max(x[0] for x in diffs), 4), "max_abs": max(x[1] for x in diffs)}
            print(log.log_id[:8], mode, r[f"match_{mode}"], flush=True)
        # 3. timing
        t0s_10 = [int(x) for x in ok[ok <= ok[0] + 13_000_000_000][::2]]  # 0.1 s steps (every other 20 Hz frame)
        t0s_1 = [int(x) for x in ok[ok <= ok[0] + 13_000_000_000][::20]]  # 1 s steps
        for mode in ["old"] + modes:
            for name, seq in (("hz10", t0s_10), ("hz1", t0s_1)):
                if mode == "old":
                    src = OldIncremental(log)
                    fn = src.sample
                else:
                    src = RF.FastRenderer(log, "wide_ft", decode=mode)
                    fn = lambda t, s=src: s.sample(t, "x")  # noqa: E731
                ms = [synced(lambda t=t: fn(t))[1] for t in seq]
                r[f"{mode}_{name}"] = {"first_ms": round(ms[0], 1), "median_ms": round(float(np.median(ms[3:])), 1),
                                       "p90_ms": round(float(np.percentile(ms[3:], 90)), 1),
                                       "max_ms": round(float(np.max(ms[3:])), 1), "n": len(ms) - 3}
                print(log.log_id[:8], mode, name, r[f"{mode}_{name}"], flush=True)
        res["logs"][log.log_id] = r
    Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

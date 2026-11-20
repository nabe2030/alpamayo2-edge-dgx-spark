#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_render_bench.py <log_dir> <out.json> [config]
# Measures, on its own, the re-rendering into virtual cameras (processing not included in the inference time).
#   1. Time to build the remap tables (maps): only once as long as the calibration does not change. For the 6 slot types
#   2. Time to render all of one input (22 images) without table lookup: 20 t0 points, 2 runs each (the 1st is discarded)
#      Breakdown: JPEG loading (PIL, CPU) and the remap on the GPU (transfer, grid_sample, until copied back to the CPU)
#   3. Time to render only the increment for each new t0 in continuous input (images from the previous t0 are reused): 10Hz and 1Hz
# None of these run concurrently with inference (the GPU is used by one process only).
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import av2_build as B  # noqa: E402


def sync_time(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    r = fn()
    torch.cuda.synchronize()
    return r, (time.perf_counter() - t) * 1000


def main():
    log = B.AV2Log(sys.argv[1])
    cfg = B.CONFIGS[sys.argv[3] if len(sys.argv) > 3 else "wide_ft"]
    res = {"log_id": log.log_id, "config": sys.argv[3] if len(sys.argv) > 3 else "wide_ft"}
    torch.zeros(1, device="cuda")
    # 1. build the tables
    t = time.perf_counter()
    for cam in (0, 1, 2, 3, 5, 6):
        B.maps_for(log, cfg[cam])
    res["maps_build_s"] = round(time.perf_counter() - t, 2)
    fc = log.cams["ring_front_center"]["ts"]
    lo = max(int(c["ts"][0]) for c in log.cams.values()) + 2_000_000_000
    t0s = [int(x) for x in fc[fc >= lo][::6][:20]]
    # 2. render all 22 images (without table lookup). Also measure the time of JPEG loading alone separately
    full, jpeg = [], []
    for rep in range(2):
        for t0 in t0s:
            _, ms = sync_time(lambda: B.build_sample(log, t0, cfg, "x"))
            files = set()
            for cam in (0, 1, 2, 3, 5, 6):
                offs = B.FRONT_OFFSETS_US if cam in (1, 6) else B.OTHER_OFFSETS_US
                for o in offs:
                    for c, _, _ in B.maps_for(log, cfg[cam]):
                        files.add((c, log.frame_at_or_before(c, t0 + o * 1000)))
            t = time.perf_counter()
            for c, ts in files:
                log.image(c, ts)
            jms = (time.perf_counter() - t) * 1000
            if rep == 1:
                full.append(ms)
                jpeg.append(jms)
    res["full_22img_ms"] = {"median": round(float(np.median(full)), 1), "max": round(float(np.max(full)), 1), "n": len(full)}
    res["jpeg_decode_ms_in_full"] = {"median": round(float(np.median(jpeg)), 1), "files_per_t0": len(files)}
    # 3. continuous input: reuse the images of the previous t0 and render only the increment
    for name, step in (("stream10hz", 2), ("stream1hz", 20)):  # front-center frames are at 50 ms intervals
        pool, per = {}, []

        def rnd(lg, spec, t_req, device):
            maps = B.maps_for(lg, spec)
            k = (json.dumps(spec, sort_keys=True), tuple(lg.frame_at_or_before(c, t_req) for c, _, _ in maps))
            if k not in pool:
                pool[k] = B.render(lg, spec, t_req, device)
            return pool[k]

        seq = [int(x) for x in fc[fc >= lo][::step]][:40]
        for i, t0 in enumerate(seq):
            n0 = len(pool)
            _, ms = sync_time(lambda: B.build_sample(log, t0, cfg, "x", render_fn=rnd))
            if i >= 3:  # not counted at the start, since there are no previous images yet
                per.append((ms, len(pool) - n0))
        res[name] = {"per_t0_ms_median": round(float(np.median([p[0] for p in per])), 1),
                     "per_t0_ms_max": round(float(np.max([p[0] for p in per])), 1),
                     "new_images_per_t0": int(np.median([p[1] for p in per])), "n": len(per)}
    Path(sys.argv[2]).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

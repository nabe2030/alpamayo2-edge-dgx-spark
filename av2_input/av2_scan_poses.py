#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_scan_poses.py <split> <out_dir>
# For every log in the split, fetch only the poses (city_SE3_egovehicle.feather, about 0.2 MB) and the map file name,
# and list the driving behavior (speed, heading change, stopped time) and city. Material for scene selection (cameras are not fetched).
import concurrent.futures as cf
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
from av2_list import BASE, list_ids_and_map  # noqa: E402


def one(split, log, out):
    p = out / f"{log}.feather"
    if not p.exists():
        with urllib.request.urlopen(f"{BASE}datasets/av2/sensor/{split}/{log}/city_SE3_egovehicle.feather", timeout=60) as r:
            p.with_suffix(".tmp").write_bytes(r.read())
        p.with_suffix(".tmp").rename(p)
    d = pd.read_feather(p)
    t = (d.timestamp_ns.values - d.timestamp_ns.values[0]) / 1e9
    xy = d[["tx_m", "ty_m"]].values
    yaw = np.unwrap(R.from_quat(d[["qx", "qy", "qz", "qw"]].values).as_euler("zyx")[:, 0])
    # resample to 0.1 s steps and compute speed and heading change
    g = np.arange(0, t[-1], 0.1)
    gx, gy, gyaw = (np.interp(g, t, v) for v in (xy[:, 0], xy[:, 1], yaw))
    v = np.hypot(np.diff(gx), np.diff(gy)) / 0.1
    return {"log_id": log, "dur_s": round(t[-1], 2), "dist_m": round(float(v.sum() * 0.1), 1),
            "v_mean": round(float(v.mean()), 2), "v_max": round(float(v.max()), 2),
            "stop_s": round(float((v < 0.5).sum() * 0.1), 1),
            "yaw_net_deg": round(float(np.degrees(gyaw[-1] - gyaw[0])), 1),
            "yaw_absmax_deg": round(float(np.degrees(np.abs(gyaw - gyaw[0]).max())), 1)}


if __name__ == "__main__":
    split, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    ids = list_ids_and_map(split)
    rows = []
    with cf.ThreadPoolExecutor(16) as ex:
        futs = {ex.submit(one, split, log, out): (log, city) for log, city in ids}
        for f in cf.as_completed(futs):
            r = f.result()
            r["city"] = futs[f][1]
            rows.append(r)
    df = pd.DataFrame(rows).sort_values("log_id")
    df.to_csv(out / "scan.csv", index=False)
    print(df.describe().round(1).to_string())
    print(df.city.value_counts().to_string())

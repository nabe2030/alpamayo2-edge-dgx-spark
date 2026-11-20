#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_select_wide.py <val_poses_dir> <train_poses_dir> <out_csv> [n_train_per_turn] [seed]
# Selects the wide set of scenes (about 400) from poses only.
#   val: up to 2 scenes from each of all 150 logs. The first is the t0 with the largest heading change among turning t0s if any,
#        otherwise random among candidates. The second is at least 3 s from the first, preferably a t0 with a different behavior (random).
#   train: to add more turning scenes, n logs that have a left t0 and n logs that have a right t0
#          (logs do not overlap). t0 is the one with the largest heading change.
# Behavior classes are the same as av2_select.py (6.4 s after t0: left/right over 30 degrees, straight under 10 degrees and 3 m/s or more,
# stop mean under 0.5 m/s, other otherwise). t0 candidates run from 2.5 s after log start to 6.5 s before the end, in 0.5 s steps
# (0.5 s further inside than before, so that 2 s of history is available even if the cameras start later than the poses).
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def candidates(poses: Path, split: str) -> pd.DataFrame:
    scan = pd.read_csv(poses / "scan.csv").set_index("log_id")
    rows = []
    for log in scan.index:
        d = pd.read_feather(poses / f"{log}.feather")
        t = (d.timestamp_ns.values - d.timestamp_ns.values[0]) / 1e9
        yaw = np.unwrap(R.from_quat(d[["qx", "qy", "qz", "qw"]].values).as_euler("zyx")[:, 0])
        for t0 in np.arange(2.5, t[-1] - 6.5 - 0.05, 0.5):
            g = t0 + np.arange(0, 6.41, 0.1)
            x, y, a = (np.interp(g, t, v) for v in (d.tx_m.values, d.ty_m.values, yaw))
            v = np.hypot(np.diff(x), np.diff(y)).sum() / 6.4
            dyaw = np.degrees(a[-1] - a[0])
            kind = ("stop" if v < 0.5 else "left" if dyaw > 30 else "right" if dyaw < -30
                    else "straight" if abs(dyaw) < 10 and v > 3 else "other")
            rows.append({"split": split, "log_id": log, "t0_rel_s": round(float(t0), 1), "kind": kind,
                         "v_mean": round(float(v), 2), "dyaw_deg": round(float(dyaw), 1), "city": scan.loc[log, "city"]})
    return pd.DataFrame(rows)


def main():
    vp, tp, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    n_train = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    rng = np.random.default_rng(int(sys.argv[5]) if len(sys.argv) > 5 else 20261119)
    picked = []
    val = candidates(vp, "val")
    for log, c in val.groupby("log_id", sort=True):
        turns = c[c.kind.isin(["left", "right"])]
        first = turns.loc[turns.dyaw_deg.abs().idxmax()] if len(turns) else c.iloc[rng.integers(len(c))]
        picked.append(first)
        far = c[(c.t0_rel_s - first.t0_rel_s).abs() >= 3.0]
        if len(far):
            other = far[far.kind != first.kind]
            pool = other if len(other) else far
            picked.append(pool.iloc[rng.integers(len(pool))])
    train = candidates(tp, "train")
    used = set()
    for kind in ("left", "right"):
        logs = np.array(sorted(set(train[train.kind == kind].log_id) - used))
        rng.shuffle(logs)
        for log in logs[:n_train]:
            c = train[(train.log_id == log) & (train.kind == kind)]
            picked.append(c.loc[c.dyaw_deg.abs().idxmax()])
            used.add(log)
    sel = pd.DataFrame(picked).reset_index(drop=True)
    sel.insert(0, "scene_id", [f"{r.split[0]}_{r.log_id[:8]}_{r.t0_rel_s:.1f}" for r in sel.itertuples()])
    sel.to_csv(out, index=False)
    print(len(sel), "scenes")
    print(sel.groupby(["split", "kind"]).size().unstack(fill_value=0).to_string())
    print(sel.city.value_counts().to_string())


if __name__ == "__main__":
    main()

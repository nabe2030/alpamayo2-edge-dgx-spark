#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_select_wide.py <val_poses_dir> <train_poses_dir> <out_csv> [n_train_per_turn] [seed]
#  の広い場面(約 400)を、姿勢だけから選ぶ。
#   val: 150 本すべてから最大 2 場面ずつ。1 つ目は曲がる t0 があればその中で向きの変化が最大の t0、
#        無ければ候補からランダム。2 つ目は 1 つ目から 3 s 以上離れ、なるべく別の走り方の t0(ランダム)。
#   train: 曲がる場面を増やすため、left の t0 があるログから n 本、right の t0 があるログから n 本
#          (ログは重ならない)。t0 は向きの変化が最大の t0。
# 走り方の分け方は av2_select.py と同じ(t0 の先 6.4 s: left/right は 30 度超、straight は 10 度未満で 3 m/s 以上、
# stop は平均 0.5 m/s 未満、other はそれ以外)。t0 の候補はログの先頭から 2.5 s 〜 末尾の 6.5 s 前、0.5 s きざみ
# (カメラの始まりが姿勢より遅れても 2 s の履歴が取れるよう、 より 0.5 s 内側から)。
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
    print(len(sel), "場面")
    print(sel.groupby(["split", "kind"]).size().unstack(fill_value=0).to_string())
    print(sel.city.value_counts().to_string())


if __name__ == "__main__":
    main()

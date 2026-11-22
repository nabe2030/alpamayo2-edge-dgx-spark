#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
"""Pick AV2 train logs that are NOT used by the 400-scene evaluation set, for FP4 calibration.

Output CSV has the columns av2_fetch_frames.py expects (log_id, t0_rel_s, kind, v_mean, dyaw_deg, city, split).
One scene per log, t0_rel_s = 8.0 s (2 s of history and 6.4 s of future fit in a ~16 s log).
Motion kinds are balanced roughly from the scan (net yaw, mean speed, stop time).
"""
import argparse, csv, random


def kind(r):
    y, v, stop = float(r["yaw_net_deg"]), float(r["v_mean"]), float(r["stop_s"])
    if y > 45:
        return "left"
    if y < -45:
        return "right"
    if stop > 3 or v < 1.5:
        return "stop"
    if abs(y) < 10 and v > 4:
        return "straight"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--eval-scenes", required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    used = {r["log_id"] for r in csv.DictReader(open(a.eval_scenes))}
    rows = [r for r in csv.DictReader(open(a.scan)) if r["log_id"] not in used and float(r["dur_s"]) > 15.0]
    rng = random.Random(a.seed)
    rng.shuffle(rows)
    by = {}
    for r in rows:
        by.setdefault(kind(r), []).append(r)
    quota = {"straight": 0.3, "left": 0.2, "right": 0.2, "stop": 0.15, "other": 0.15}
    pick = []
    for k, q in quota.items():
        pick += by.get(k, [])[: round(q * a.n)]
    rest = [r for r in rows if r not in pick]
    pick += rest[: a.n - len(pick)]
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["log_id", "t0_rel_s", "kind", "v_mean", "dyaw_deg", "city", "split"])
        for r in pick[: a.n]:
            w.writerow([r["log_id"], 8.0, kind(r), r["v_mean"], r["yaw_net_deg"], r["city"], "train"])
    print(len(pick[: a.n]), "logs;", {k: sum(kind(r) == k for r in pick[: a.n]) for k in quota}, "; eval logs excluded:", len(used))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_fetch_frames.py <scenes_csv> <dest_root> <full_val_dir> <out_csv> [workers]
# For each scene, fetch only the frames needed to build the input from public S3 (anonymous HTTPS).
#   Per log: the 2 files in calibration/, city_SE3_egovehicle.feather, map/log_map_archive_*.json (for intersection detection in the navigation text)
#   Per scene: for each of the 7 ring cameras × t0 + {−2, −1, −0.5, −0.1, 0} s, the frame "at or before (1 ms tolerance)"
#     (= enough for every layout; wide_ft uses a subset of these)
# t0 is determined as in av2_build.AV2Log.t0_from_rel, from the list of all camera frames (S3 listing); t0_ns and data_dir are written to out_csv.
# Logs already fetched in full (<full_val_dir>/<log_id>) are not fetched again; that directory is used.
# Location: <dest_root>/<split>/<log_id>/ (same layout as the logs). Per-file sha256 and source URL are recorded in manifest.json.
import concurrent.futures as cf
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from av2_fetch import get  # noqa: E402
from av2_list import BASE, list_objects  # noqa: E402

RING = ("ring_front_center", "ring_front_left", "ring_front_right", "ring_side_left",
        "ring_side_right", "ring_rear_left", "ring_rear_right")
OFFS_NS = [int(o * 1e9) for o in (-2.0, -1.0, -0.5, -0.1, 0.0)]
TOL_NS = 1_000_000


def at_or_before(ts, t):
    i = np.searchsorted(ts, t + TOL_NS, side="right") - 1
    if i < 0:
        raise ValueError("no earlier frame")
    return int(ts[i])


def plan_log(split, log, t0_rels, root):
    """The list of files to fetch for one log, and t0_ns per scene."""
    keys, _ = list_objects(f"datasets/av2/sensor/{split}/{log}/")
    rel = {k.split(f"{log}/", 1)[1]: s for k, s in keys}
    ts = {c: np.array(sorted(int(Path(k).stem) for k in rel if k.startswith(f"sensors/cameras/{c}/")), dtype=np.int64)
          for c in RING}
    need = {k: s for k, s in rel.items() if k.startswith("calibration/") or k == "city_SE3_egovehicle.feather"
            or (k.startswith("map/log_map_archive_") and k.endswith(".json"))}
    # fetch the pose file first to know where the poses end (upper limit of t0)
    d = root / f"{log}.partial"
    (d).mkdir(parents=True, exist_ok=True)
    pose = d / "city_SE3_egovehicle.feather"
    if not pose.exists():
        get(f"{BASE}datasets/av2/sensor/{split}/{log}/city_SE3_egovehicle.feather", pose)
    pts = pd.read_feather(pose).timestamp_ns.values
    fc = ts["ring_front_center"]
    origin = int(fc[0])
    lo = max(int(v[0]) for v in ts.values()) + 2_000_000_000
    hi = int(pts.max()) - 6_400_000_000
    ok = fc[(fc >= lo) & (fc <= hi)]
    t0s = []
    for r in t0_rels:
        target = origin + int(round(r * 1e9))
        t0 = int(ok[np.argmin(np.abs(ok - target))])
        t0s.append(t0)
        for c in RING:
            for o in OFFS_NS:
                k = f"sensors/cameras/{c}/{at_or_before(ts[c], t0 + o)}.jpg"
                need[k] = rel[k]
    return need, t0s


def fetch_log(split, log, t0_rels, root):
    final = root / log
    man_path = final / "manifest.json"
    prev = json.loads(man_path.read_text()) if man_path.exists() else None
    need, t0s = plan_log(split, log, t0_rels, root)
    part = root / f"{log}.partial"
    if final.exists():  # when adding scenes to a previously fetched log: move its contents back to partial and add
        for p in final.rglob("*"):
            if p.is_file():
                q = part / p.relative_to(final)
                q.parent.mkdir(parents=True, exist_ok=True)
                p.rename(q)
        for p in sorted(final.rglob("*"), reverse=True):
            p.rmdir()
        final.rmdir()
    files = dict(prev["files"]) if prev else {}
    todo = []
    for k, s in need.items():
        p = part / k
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and p.stat().st_size == s:
            if k not in files:
                files[k] = {"bytes": s, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            continue
        todo.append((k, s, p))
    with cf.ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(get, f"{BASE}datasets/av2/sensor/{split}/{log}/{k}", p): (k, s) for k, s, p in todo}
        for f in cf.as_completed(futs):
            k, s = futs[f]
            files[k] = {"bytes": s, "sha256": f.result()}
    for k, s in need.items():
        assert (part / k).stat().st_size == s, k
    (part / "manifest.json").write_text(json.dumps({
        "dataset": "Argoverse 2 Sensor Dataset (CC BY-NC-SA 4.0, © 2021 Argo AI, LLC)", "split": split, "log_id": log,
        "source": f"{BASE}datasets/av2/sensor/{split}/{log}/", "fetched": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "partial": "needed frames only (av2_fetch_frames.py)", "files": files}, indent=1))
    part.rename(final)
    return t0s, sum(s for _, s, _ in todo)


def main():
    sc = pd.read_csv(sys.argv[1])
    root, full, out = Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    sc["t0_ns"] = 0
    sc["data_dir"] = ""
    total, t = 0, time.time()
    groups = list(sc.groupby(["split", "log_id"], sort=True))
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 6

    def one(item):
        (split, log), g = item
        if split == "val" and (full / log).is_dir():
            # log fetched in full: t0 is determined the same way as av2_build
            import av2_build as B
            lg = B.AV2Log(full / log)
            return g.index, [lg.t0_from_rel(r) for r in g.t0_rel_s], str(full / log), 0
        d = root / split
        d.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                t0s, b = fetch_log(split, log, list(g.t0_rel_s), d)
                return g.index, t0s, str(d / log), b
            except Exception as e:  # transient failures are retried per log (what was fetched is kept)
                print(f"  retry {log} {type(e).__name__}: {e}", flush=True)
                if attempt == 2:
                    raise
                time.sleep(20)

    with cf.ThreadPoolExecutor(workers) as ex:  # logs in parallel (one log is dominated by listing and round-trip waits)
        for n, (idx, t0s, ddir, b) in enumerate(ex.map(one, groups)):
            total += b
            sc.loc[idx, "t0_ns"] = t0s
            sc.loc[idx, "data_dir"] = ddir
            if n % 20 == 0:
                print(f"[{n + 1}/{len(groups)}] total {total / 1e9:.2f} GB {time.time() - t:.0f}s", flush=True)
    sc.to_csv(out, index=False)
    print(f"done: {len(groups)} logs, fetched {total / 1e9:.2f} GB, {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()

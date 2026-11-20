#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_fetch.py <split> <dest_root> <log_id> [<log_id> ...]
# AV2 Sensor の公開バケット(匿名の HTTPS)から、入力の組み立てに要るものだけを取る:
#   リングカメラ 7 台(sensors/cameras/ring_*)、calibration/、city_SE3_egovehicle.feather、map/、annotations.feather
#   (LiDAR とステレオカメラは取らない)
# <dest_root>/<log_id>.partial に書いてから名前を変える。ファイルごとの sha256 と元の URL を manifest.json に残す。
import concurrent.futures as cf
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from av2_list import BASE, list_objects  # noqa: E402

WANT = ("sensors/cameras/ring_", "calibration/", "city_SE3_egovehicle.feather", "map/", "annotations.feather")


def get(url, path, tries=4):
    for i in range(tries):
        try:
            h = hashlib.sha256()
            tmp = path.with_suffix(path.suffix + ".tmp")
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    h.update(chunk)
                    f.write(chunk)
            tmp.rename(path)
            return h.hexdigest()
        except Exception as e:  # 一時的な失敗はやり直す
            if i == tries - 1:
                raise
            print(f"  やり直し {i + 1}: {url} {type(e).__name__}", flush=True)
            time.sleep(10 * (i + 1))


def fetch_log(split, root, log):
    final = root / log
    if final.exists():
        print(f"{log}: 取得済み", flush=True)
        return 0
    part = root / f"{log}.partial"
    keys, _ = list_objects(f"datasets/av2/sensor/{split}/{log}/")
    todo = [(k, s) for k, s in keys if k.split(f"{log}/", 1)[1].startswith(WANT)]
    total = sum(s for _, s in todo)
    print(f"{log}: {len(todo)} 件 {total / 1e9:.3f} GB", flush=True)
    t = time.time()
    files = {}
    with cf.ThreadPoolExecutor(16) as ex:
        futs = {}
        for k, s in todo:
            rel = k.split(f"{log}/", 1)[1]
            p = part / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists() and p.stat().st_size == s:
                continue
            futs[ex.submit(get, BASE + k, p)] = (rel, s)
        for f in cf.as_completed(futs):
            rel, s = futs[f]
            files[rel] = {"bytes": s, "sha256": f.result()}
    for k, s in todo:  # 前回の途中から再開した分も含め、全件を照合
        rel = k.split(f"{log}/", 1)[1]
        p = part / rel
        assert p.stat().st_size == s, rel
        if rel not in files:
            files[rel] = {"bytes": s, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    (part / "manifest.json").write_text(json.dumps({
        "dataset": "Argoverse 2 Sensor Dataset", "split": split, "log_id": log,
        "source": f"{BASE}datasets/av2/sensor/{split}/{log}/", "fetched": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "bytes": total, "files": files}, indent=1))
    part.rename(final)
    print(f"{log}: 完了 {time.time() - t:.0f}s", flush=True)
    return total


if __name__ == "__main__":
    split, root = sys.argv[1], Path(sys.argv[2])
    root.mkdir(parents=True, exist_ok=True)
    tot = sum(fetch_log(split, root, log) for log in sys.argv[3:])
    print(f"合計 {tot / 1e9:.3f} GB")

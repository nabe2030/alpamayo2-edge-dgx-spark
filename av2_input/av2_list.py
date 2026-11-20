#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_list.py — AV2 の公開バケット(s3://argoverse、匿名で読める)を HTTPS の一覧 API で数える。中身は取らない。
#   av2_list.py logs <split> [<n>]        … split のログ ID を n 本(省略時は全部)
#   av2_list.py sizes <split> <log_id>    … 1 本のログのファイルを種類ごとに数え、容量を出す
#   av2_list.py splitsize <split> [<n>]   … split のログを先頭から n 本測り、合計と 1 本あたりを出す
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict

BASE = "https://argoverse.s3.us-east-1.amazonaws.com/"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_objects(prefix, delimiter=None):
    """一覧 API を続きのトークンでたどり、(キー, サイズ) と共通の接頭辞を返す。"""
    token, keys, prefixes = None, [], []
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if delimiter:
            q["delimiter"] = delimiter
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(BASE + "?" + urllib.parse.urlencode(q), timeout=60) as r:
            root = ET.fromstring(r.read())
        for c in root.findall("s:Contents", NS):
            keys.append((c.find("s:Key", NS).text, int(c.find("s:Size", NS).text)))
        for p in root.findall("s:CommonPrefixes", NS):
            prefixes.append(p.find("s:Prefix", NS).text)
        if root.find("s:IsTruncated", NS).text != "true":
            return keys, prefixes
        token = root.find("s:NextContinuationToken", NS).text


def log_ids(split):
    _, prefixes = list_objects(f"datasets/av2/sensor/{split}/", "/")
    return [p.rstrip("/").split("/")[-1] for p in prefixes]


def sizes(split, log):
    keys, _ = list_objects(f"datasets/av2/sensor/{split}/{log}/")
    groups = defaultdict(lambda: [0, 0])
    for k, s in keys:
        rel = k.split(f"{log}/", 1)[1]
        parts = rel.split("/")
        g = "/".join(parts[:3]) if parts[0] == "sensors" and parts[1] == "cameras" else (
            "/".join(parts[:2]) if len(parts) > 2 else rel)
        groups[g][0] += 1
        groups[g][1] += s
    return dict(groups)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "logs":
        ids = log_ids(sys.argv[2])
        n = int(sys.argv[3]) if len(sys.argv) > 3 else len(ids)
        print(len(ids), "本")
        print("\n".join(ids[:n]))
    elif cmd == "sizes":
        g = sizes(sys.argv[2], sys.argv[3])
        tot = 0
        for k in sorted(g):
            print(f"{k:60s} {g[k][0]:6d} 件 {g[k][1]/1e6:10.1f} MB")
            tot += g[k][1]
        print(f"合計 {tot/1e9:.3f} GB")
    elif cmd == "splitsize":
        ids = log_ids(sys.argv[2])
        n = int(sys.argv[3]) if len(sys.argv) > 3 else len(ids)
        tot = cam = 0
        for i in ids[:n]:
            g = sizes(sys.argv[2], i)
            t = sum(v[1] for v in g.values())
            c = sum(v[1] for k, v in g.items() if k.startswith("sensors/cameras/ring_"))
            tot += t
            cam += c
        print(f"{sys.argv[2]}: 全 {len(ids)} 本、先頭 {n} 本の平均 全体 {tot/n/1e9:.3f} GB/本、リングカメラ {cam/n/1e9:.3f} GB/本")


def list_ids_and_map(split):
    """ログ ID と、地図のファイル名から読める都市の略号(PIT、MIA など)の組を返す。"""
    out = []
    for log in log_ids(split):
        keys, _ = list_objects(f"datasets/av2/sensor/{split}/{log}/map/")
        city = next((k.split("____")[1].split("_")[0] for k, _ in keys if "log_map_archive" in k), "?")
        out.append((log, city))
    return out

#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_list.py — counts the public AV2 bucket (s3://argoverse, anonymously readable) with the HTTPS listing API. Does not fetch contents.
#   av2_list.py logs <split> [<n>]        … n log IDs of the split (all if omitted)
#   av2_list.py sizes <split> <log_id>    … counts the files of one log by type and prints their sizes
#   av2_list.py splitsize <split> [<n>]   … measures the first n logs of the split and prints the total and per-log size
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict

BASE = "https://argoverse.s3.us-east-1.amazonaws.com/"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_objects(prefix, delimiter=None):
    """Follow the listing API with continuation tokens and return (key, size) pairs and common prefixes."""
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
        print(len(ids), "logs")
        print("\n".join(ids[:n]))
    elif cmd == "sizes":
        g = sizes(sys.argv[2], sys.argv[3])
        tot = 0
        for k in sorted(g):
            print(f"{k:60s} {g[k][0]:6d} files {g[k][1]/1e6:10.1f} MB")
            tot += g[k][1]
        print(f"total {tot/1e9:.3f} GB")
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
        print(f"{sys.argv[2]}: {len(ids)} logs in all; average of the first {n}: total {tot/n/1e9:.3f} GB/log, ring cameras {cam/n/1e9:.3f} GB/log")


def list_ids_and_map(split):
    """Return pairs of log ID and city code read from the map file name (PIT, MIA, etc.)."""
    out = []
    for log in log_ids(split):
        keys, _ = list_objects(f"datasets/av2/sensor/{split}/{log}/map/")
        city = next((k.split("____")[1].split("_")[0] for k, _ in keys if "log_map_archive" in k), "?")
        out.append((log, city))
    return out

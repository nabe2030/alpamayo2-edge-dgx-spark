#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
# cmp_greedy.py <ref_dir> <test_dir> <label> — greedy CoT of a PyTorch variant (greedy_fakeq.py) vs the BF16
# references (tensorrt/trt_ref_multi.py), with the same yardsticks as tensorrt/trt_eval_multi.py: identical CoT, top-1 at step 0,
# per-step KL over the BF16 top-20 (renormalized) while both agree.
import glob, json, os, sys
import numpy as np

ref_dir, test_dir, label = sys.argv[1:4]
res = []
for d in sorted(glob.glob(os.path.join(ref_dir, "*", "ref.json"))):
    sid = os.path.basename(os.path.dirname(d))
    r, t = json.load(open(d)), json.load(open(os.path.join(test_dir, sid, "ref.json")))
    g, o = r["greedy_ids"], t["greedy_ids"]
    pre = 0
    for x, y in zip(o, g):
        if x != y:
            break
        pre += 1
    kls = []
    for s in range(min(pre + 1, len(g), len(o))):
        ids, vals = r["greedy_top20"][s]
        ti, tv = t["greedy_top20"][s]
        tm = dict(zip(ti, tv))
        floor = min(tv) - 2.0
        p = np.exp(np.array(vals)); p /= p.sum()
        q = np.exp(np.array([tm.get(i, floor) for i in ids])); q /= q.sum()
        kls.append(float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)))))
    res.append({"scene": sid, "identical": o == g, "top1_0": bool(o and g and o[0] == g[0]), "kls": kls})
allkl = [k for x in res for k in x["kls"]]
print(json.dumps({"label": label, "scenes": len(res), "identical": sum(x["identical"] for x in res),
                  "top1_step0_same": sum(x["top1_0"] for x in res), "kl_median": float(np.median(allkl)),
                  "kl_mean": float(np.mean(allkl)), "kl_steps": len(allkl)}))

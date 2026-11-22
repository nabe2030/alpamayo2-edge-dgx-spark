#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
# fp4_qerr.py — compare FP4 rounding methods on the real Alpamayo 2 Edge text-layer weights.
#
# For every projection (qkv / o / fc1 / fc2 of the 28 layers) and method, report
#   plain error   : ||W - Wq||_F^2 / ||W||_F^2
#   output error  : tr(dW H dW^T) / tr(W H W^T)   (H = sum x x^T on the AV2 calibration decode steps)
# and, with --held-out, the output error measured with a second H collected on different scenes.
# Quantized bodies are saved in the a2e_fp4q cache format (A2E_FP4_CACHE/body_<method>.pt).
import argparse
import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open

import a2e_fp4q as A
import a2e_fp4 as F


def load_weights(model_dir, n_layers=28):
    idx = json.load(open(Path(model_dir) / "model.safetensors.index.json"))["weight_map"]
    handles = {}

    def get(k):
        fn = idx[k]
        if fn not in handles:
            handles[fn] = safe_open(str(Path(model_dir) / fn), framework="pt", device="cuda")
        return handles[fn].get_tensor(k)

    pre = "vlm.model.language_model.layers"
    out = {}
    for l in range(n_layers):
        p = f"{pre}.{l}"
        out[l] = {"qkv": torch.cat([get(f"{p}.self_attn.q_proj.weight"), get(f"{p}.self_attn.k_proj.weight"),
                                    get(f"{p}.self_attn.v_proj.weight")], 0),
                  "o": get(f"{p}.self_attn.o_proj.weight"), "fc1": get(f"{p}.mlp.fc1.weight"),
                  "fc2": get(f"{p}.mlp.fc2.weight")}
    return out


def out_err(W, Wq, H):
    d = (W - Wq).float()
    Wf = W.float()
    return float(((d @ H) * d).sum() / ((Wf @ H) * Wf).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True, help="H used for calibration (a2e_fp4q CalibStats file)")
    ap.add_argument("--held-out", default="", help="H from other scenes, used only for measuring")
    ap.add_argument("--methods", default="rtn,mseu,mse,gptq,gptqm")
    ap.add_argument("--layers", default="", help="subset, e.g. 0,13,27 (quick look; no cache saved)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["A2E_CALIB_IN"] = a.calib
    A.CALIB_IN = a.calib
    Hc = A.calib_H()
    Hh = torch.load(a.held_out, map_location="cpu")["H"] if a.held_out else None
    W = load_weights(a.model)
    layers = [int(x) for x in a.layers.split(",")] if a.layers else list(range(28))
    res = []
    for m in a.methods.split(","):
        t = time.time()
        body = {}
        for l in layers:
            body[l] = {}
            for k, w in W[l].items():
                key = f"{l}.{k}"
                q = A.quantize(m, w, key)
                body[l][k] = q
                wq = F.dequantize_nvfp4(*q)
                Hk = Hc[key].cuda()
                r = {"method": m, "layer": l, "proj": k,
                     "plain": float(((w.float() - wq) ** 2).sum() / (w.float() ** 2).sum()),
                     "out_calib": out_err(w, wq, Hk)}
                if Hh is not None:
                    r["out_held"] = out_err(w, wq, Hh[key].cuda())
                res.append(r)
        dt = time.time() - t
        sel = [r for r in res if r["method"] == m]
        line = {k: sum(r[k] for r in sel) / len(sel) for k in ("plain", "out_calib", "out_held") if k in sel[0]}
        print(f"{m}: {dt:.0f}s mean " + " ".join(f"{k}={v:.4%}" for k, v in line.items()), flush=True)
        if not a.layers and A.CACHE_DIR:
            p = Path(A.CACHE_DIR) / f"body_{m}.pt"
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(body, p)
            print(f"  saved {p}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()

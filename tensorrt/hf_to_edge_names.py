#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
"""Rename a Transformers-layout Cosmos3-Edge checkpoint (e.g. ModelOpt NVFP4 output) to the names the
Edge-LLM direct builder resolves: model.language_model.X -> model.X, mlp.fc1/fc2 -> mlp.up_proj/down_proj
(language model only; the SigLIP2 tower keeps fc1/fc2). Tensors are copied byte-for-byte; every
non-safetensors file (configs, hf_quant_config.json, tokenizer) is copied as is.
"""
import argparse, json, os, shutil, struct


def rename(k):
    if k.startswith("model.language_model."):
        k = "model." + k[len("model.language_model."):]
        return k.replace(".mlp.fc1.", ".mlp.up_proj.").replace(".mlp.fc2.", ".mlp.down_proj.")
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    wmap = {}
    for fn in sorted(os.listdir(a.src)):
        sp = os.path.join(a.src, fn)
        if not fn.endswith(".safetensors"):
            if os.path.isfile(sp) and fn != "model.safetensors.index.json":
                shutil.copy(sp, a.dst)
            continue
        with open(sp, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        off = 8 + n
        meta = hdr.pop("__metadata__", None)
        items = sorted(((rename(k), k, v) for k, v in hdr.items()), key=lambda t: t[2]["data_offsets"][0])
        nh, pos = {}, 0
        for nk, k, v in items:
            s, e = v["data_offsets"]
            nh[nk] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [pos, pos + e - s]}
            pos += e - s
        if meta:
            nh["__metadata__"] = meta
        hb = json.dumps(nh, separators=(",", ":")).encode()
        hb += b" " * ((8 - len(hb) % 8) % 8)
        with open(sp, "rb") as fi, open(os.path.join(a.dst, fn), "wb") as fo:
            fo.write(struct.pack("<Q", len(hb))); fo.write(hb)
            for nk, k, v in items:
                s, e = v["data_offsets"]
                fi.seek(off + s); left = e - s
                while left:
                    b = fi.read(min(left, 64 << 20)); fo.write(b); left -= len(b)
                wmap[nk] = fn
        print(fn, len(items))
    json.dump({"metadata": {}, "weight_map": wmap}, open(os.path.join(a.dst, "model.safetensors.index.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

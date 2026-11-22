#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
"""Extract the VLM (Cosmos3-Edge reasoner) part of nvidia/Alpamayo2-Edge into a
Cosmos3-Edge style checkpoint that TensorRT Edge-LLM can consume.

- Keys:  vlm.model.language_model.X -> model.X (mlp.fc1/fc2 -> mlp.up_proj/down_proj) ; vlm.model.visual.X -> model.visual.X ;
         vlm.model.projector.X -> model.projector.X ; vlm.lm_head.weight -> lm_head.weight
- expert.* (the action expert) is dropped.
- Tensors are copied byte-for-byte (BF16 kept as is), no numerics change.
- config.json is the checkpoint's vlm_config (model_type cosmos3_edge).
"""
import argparse, json, os, shutil, struct


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


HF_MODE = False


def rename(k):
    if HF_MODE:
        # Transformers layout of Cosmos3EdgeForConditionalGeneration (for ModelOpt quantization).
        if k.startswith("vlm."):
            return k[len("vlm."):]
        if k.startswith("expert."):
            return None
        raise KeyError(k)
    if k.startswith("vlm.model.language_model."):
        k = "model." + k[len("vlm.model.language_model."):]
        # Edge-LLM's squared-ReLU MLP names the two projections up_proj / down_proj.
        return k.replace(".mlp.fc1.", ".mlp.up_proj.").replace(".mlp.fc2.", ".mlp.down_proj.")
    if k.startswith("vlm.model.visual.") or k.startswith("vlm.model.projector."):
        return k[len("vlm."):]
    if k == "vlm.lm_head.weight":
        return "lm_head.weight"
    if k.startswith("expert."):
        return None
    raise KeyError(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--hf", action="store_true", help="keep the Transformers key layout (only drop 'vlm.')")
    a = ap.parse_args()
    global HF_MODE
    HF_MODE = a.hf
    os.makedirs(a.dst, exist_ok=True)
    idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))
    files = sorted(set(idx["weight_map"].values()))
    out_map = {}
    for fn in files:
        hdr, off = read_header(os.path.join(a.src, fn))
        meta = hdr.pop("__metadata__", None)
        items = []
        for k, v in hdr.items():
            nk = rename(k)
            if nk is not None:
                items.append((nk, k, v))
        items.sort(key=lambda t: t[2]["data_offsets"][0])
        new_hdr, pos = {}, 0
        for nk, k, v in items:
            s, e = v["data_offsets"]
            new_hdr[nk] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [pos, pos + e - s]}
            pos += e - s
        if meta:
            new_hdr["__metadata__"] = meta
        hb = json.dumps(new_hdr, separators=(",", ":")).encode()
        hb += b" " * ((8 - len(hb) % 8) % 8)
        ofn = "reasoner-" + fn
        with open(os.path.join(a.src, fn), "rb") as fi, open(os.path.join(a.dst, ofn), "wb") as fo:
            fo.write(struct.pack("<Q", len(hb)))
            fo.write(hb)
            for nk, k, v in items:
                s, e = v["data_offsets"]
                fi.seek(off + s)
                left = e - s
                while left:
                    buf = fi.read(min(left, 64 << 20))
                    fo.write(buf)
                    left -= len(buf)
                out_map[nk] = ofn
        print(fn, "->", ofn, len(items), "tensors")
    json.dump({"metadata": {}, "weight_map": out_map},
              open(os.path.join(a.dst, "model.safetensors.index.json"), "w"), indent=1)
    cfg = json.load(open(os.path.join(a.src, "config.json")))["vlm_config"]
    json.dump(cfg, open(os.path.join(a.dst, "config.json"), "w"), indent=2)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "processor_config.json", "generation_config.json"):
        if os.path.exists(os.path.join(a.src, f)):
            shutil.copy(os.path.join(a.src, f), a.dst)
    print("tensors:", len(out_map))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
"""Compare Edge-LLM LayerDebugger dumps (per-layer KV cache, round 0 = prefill) of two engines.

Per layer and K/V: cosine and norm ratio over the valid prefix, plus per-token cosine statistics
(min, 1st percentile, fraction of tokens below 0.9) and the worst token positions.
"""
import argparse, json

import torch
from safetensors import safe_open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    fr = safe_open(a.ref, "pt")
    fts = [safe_open(t, "pt") for t in a.test]
    r = a.round
    valid = int(fr.get_tensor(f"round_{r}.context_lengths")[0])
    out = []
    print(f"valid={valid}")
    for i in range(a.layers):
        g = fr.get_tensor(f"round_{r}.layer_{i}.kv")[0, :, :, :valid].float().cuda()  # [2, H, S, D]
        row = {"layer": i}
        line = f"L{i:02d}"
        for ti, ft in enumerate(fts):
            e = ft.get_tensor(f"round_{r}.layer_{i}.kv")[0, :, :, :valid].float().cuda()
            for kv, name in ((0, "K"), (1, "V")):
                x, y = g[kv], e[kv]
                cos = torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0).item()
                nr = (y.norm() / x.norm()).item()
                # per token: flatten heads x dim
                xt = x.permute(1, 0, 2).reshape(valid, -1); yt = y.permute(1, 0, 2).reshape(valid, -1)
                ct = torch.nn.functional.cosine_similarity(xt, yt, dim=1)
                row[f"t{ti}_{name}"] = {"cos": cos, "norm_ratio": nr, "tok_cos_min": ct.min().item(),
                                        "tok_cos_p1": ct.quantile(0.01).item(),
                                        "frac_below_0.9": (ct < 0.9).float().mean().item(),
                                        "worst": ct.argsort()[:5].tolist()}
                line += f" | t{ti}{name} cos {cos:.4f} nr {nr:.3f} min {ct.min().item():.3f} <0.9 {(ct < 0.9).float().mean().item():.3f}"
        print(line, flush=True)
        out.append(row)
    lr = fr.get_tensor(f"round_{r}.logits")[0].float()
    for ti, ft in enumerate(fts):
        le = ft.get_tensor(f"round_{r}.logits")[0].float()
        print(f"logits t{ti}: cos {torch.nn.functional.cosine_similarity(lr, le, dim=0).item():.4f} "
              f"argmax ref {lr.argmax().item()} test {le.argmax().item()}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

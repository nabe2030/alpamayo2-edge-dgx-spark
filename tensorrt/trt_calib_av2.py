#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
"""Build AV2 calibration batches (full Alpamayo2-Edge prompts: 22 images + fused trajectory tokens)
for ModelOpt NVFP4 calibration of the Cosmos3-Edge reasoner.

Samples are AV2 logs from the 400-scene list, at t0 values at least --gap s away from every evaluated
t0 in the same log (so calibration frames are not the evaluated frames). Saved as a list of HF
BatchFeature-like dicts (input_ids, attention_mask, pixel_values fp16, image_grid_thw, mm_token_type_ids).
Run in the Alpamayo2-Edge venv after `source env.sh` (av2_input/ of this repository on PYTHONPATH).
"""
import argparse, csv, random, sys
from collections import defaultdict
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--root", default=".", help="base directory of the data_dir column of the scene csv (default: current directory, i.e. $A2E_WORK after `source env.sh`)")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--gap", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--use-t0", action="store_true",
                    help="use each row's own t0 (for a dedicated calibration scene list, e.g. the FP4 calibration logs)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    # av2_build / av2_nav come from av2_input/ of this repository (PYTHONPATH set by env.sh)
    import av2_build as B
    import av2_nav as N
    from alpamayo2_edge.checkpoint import load_model
    from alpamayo2_edge.helper import prepare_model_inputs
    from alpamayo2_edge.models.utils import fuse_traj_tokens

    rows = list(csv.DictReader(open(a.scenes)))
    evald = defaultdict(list)
    dirs, t0ns = {}, {}
    for r in rows:
        evald[r["log_id"]].append(float(r["t0_rel_s"]))
        dirs[r["log_id"]] = r["data_dir"]
        t0ns[r["log_id"]] = int(r["t0_ns"])
    logs = sorted(evald)
    random.Random(a.seed).shuffle(logs)
    model, _ = load_model(a.model, device="cuda", dtype="bfloat16", attn_implementation="sdpa", local_files_only=True)
    cfg = model.config
    batches, used = [], []
    for log in logs:
        if len(batches) >= a.n:
            break
        if a.use_t0:
            t0 = evald[log][0]
        else:
            cands = [t / 2 for t in range(5, 31)]  # 2.5 .. 15.0 s
            cands = [t for t in cands if all(abs(t - e) >= a.gap for e in evald[log])]
            if not cands:
                continue
            t0 = random.Random(hash(log) & 0xffff).choice(cands)
        try:
            # Same sample construction as measure/wide_eval.py av2_scene (absolute t0_ns).
            lg = B.AV2Log(str(Path(a.root) / dirs[log]))
            if a.use_t0:
                t0_ns = t0ns[log]
            else:
                t0_ns = lg.t_rel_origin_ns + int(t0 * 1e9)
            d = B.build_sample(lg, t0_ns, B.CONFIGS["wide_ft"], N.nav_from_future(lg, t0_ns)["text"])
            d.pop("av2", None)
        except Exception as e:  # missing frames / history: skip this log
            print("skip", log, t0, e)
            continue
        p = prepare_model_inputs(d, cfg, model.tokenizer)
        td = dict(p["tokenized_data"])
        ids = fuse_traj_tokens(model.history_traj_tokenizer, model.future_traj_tokenizer, td["input_ids"],
                               {k: p[k] for k in ("ego_history_xyz", "ego_history_rot")}, cfg.traj_ids)
        batches.append({"input_ids": ids.cpu(), "attention_mask": torch.ones_like(ids).cpu(),
                        "pixel_values": td["pixel_values"].half().cpu(), "image_grid_thw": td["image_grid_thw"].cpu(),
                        "mm_token_type_ids": td["mm_token_type_ids"].cpu()})
        used.append((log, t0, int(ids.shape[1])))
        print("ok", log, t0, ids.shape[1], flush=True)
    torch.save({"batches": batches, "used": used}, a.out)
    print("saved", len(batches))


if __name__ == "__main__":
    main()

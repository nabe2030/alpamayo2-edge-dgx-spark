#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_eval.py — on AV2 scenes, runs in BF16 as is while varying the input assembly (CONFIGS) and how the navigation text is made,
# and records the reasoning (CoT) and trajectory. ADE/FDE against the recorded drive (GT) are also computed here.
#
#   av2_eval.py --scenes raw/dev_scenes.csv --data data/av2/val --out raw/dev_eval.jsonl \
#               --configs nat,wide_ft --navs gt --seeds 43,44,45,46,47
#
# Inference uses the official path: helper.prepare_model_inputs → model.sample_trajectories_from_data
# (same settings as official inference: temperature 0.6, top_p 0.98, 1 candidate, 10 diffusion steps, diffusion RNG equal to the seed).
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
pass  # (in the working environment this line loaded scripts from another location; in the public version they are in the same folder)
import av2_build as B  # noqa: E402
import av2_nav as N  # noqa: E402


def ade_fde(p, g):
    d = np.linalg.norm(np.asarray(p)[:, :2] - np.asarray(g)[:, :2], axis=-1)
    return float(d.mean()), float(d[-1])


def const_vel(hxyz):
    """Naive prediction for comparison: keeps the speed and heading rate of the last 0.5 s and advances 6.4 s."""
    h = np.asarray(hxyz)[:, :2]
    v = (h[-1] - h[-6]) / 0.5
    a0 = np.arctan2(*(h[-6] - h[-11])[::-1]) if np.linalg.norm(h[-6] - h[-11]) > 0.05 else 0.0
    a1 = np.arctan2(*(h[-1] - h[-6])[::-1]) if np.linalg.norm(v) > 0.1 else 0.0
    w = (a1 - a0) / 0.5 if np.linalg.norm(v) > 0.5 else 0.0
    sp = np.linalg.norm(v)
    pts, x, y, a = [], 0.0, 0.0, a1
    for _ in range(64):
        a += w * 0.1
        x += sp * np.cos(a) * 0.1
        y += sp * np.sin(a) * 0.1
        pts.append([x, y, 0.0])
    return np.array(pts)


def gpu_temp():
    import subprocess
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=10).stdout.split()[0])
    except Exception:
        return -1


def thermal_gate(hot=84, resume=60, pause=180):
    """If 84°C or above, pause 3 minutes and wait until it drops to 60°C or below."""
    t = gpu_temp()
    if t >= hot:
        print(f"temperature {t}°C: pause", flush=True)
        time.sleep(pause)
        while gpu_temp() > resume:
            time.sleep(30)
        print(f"temperature {gpu_temp()}°C: resume", flush=True)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", default="nat")
    ap.add_argument("--navs", default="gt")
    ap.add_argument("--seeds", default="43,44,45,46,47")
    ap.add_argument("--model", default=str(Path(os.environ.get("A2E_MODEL", "models/Alpamayo2-Edge"))))
    ap.add_argument("--save-samples", default="", help="directory to save the 22 images at t0 (for making figures)")
    args = ap.parse_args()

    scenes = pd.read_csv(args.scenes)
    configs, navs = args.configs.split(","), args.navs.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if r.get("kind") == "run":
                done.add((r["log_id"], r["t0_rel_s"], r["config"], r["nav"], r["seed"]))

    import a2e_stagea as S
    from alpamayo2_edge.accel import AccelConfig, enable_acceleration
    from alpamayo2_edge.checkpoint import load_model
    from alpamayo2_edge.helper import prepare_model_inputs, to_device

    S.install_processor_once()
    model, _ = load_model(args.model, device="cuda", dtype="bfloat16", attn_implementation="sdpa", local_files_only=True)
    enable_acceleration(model, AccelConfig(mode="tuned"))

    f = out.open("a")

    def emit(rec):
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()

    emit({"kind": "meta", "time": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(args),
          "configs": {c: B.CONFIGS[c] for c in configs}})
    for _, sc in scenes.iterrows():
        temp = thermal_gate()
        log = B.AV2Log(Path(args.data) / sc.log_id)
        t0 = log.t0_from_rel(float(sc.t0_rel_s))
        gt = B.gt_future(log, t0)
        nav_info = N.nav_from_future(log, t0)
        hxyz, _ = log.local_traj(t0, np.arange(-1_900_000, 1, 100_000))
        cv = const_vel(hxyz)
        emit({"kind": "scene", "log_id": sc.log_id, "t0_rel_s": float(sc.t0_rel_s), "t0_ns": t0, "motion": sc.kind,
              "nav_info": nav_info, "gt": np.round(gt, 3).tolist(), "cv_ade_fde": ade_fde(cv, gt), "gpu_temp": temp})
        for cfg in configs:
            for nav in navs:
                text = N.variant(nav_info, nav)
                todo = [s for s in seeds if (sc.log_id, float(sc.t0_rel_s), cfg, nav, s) not in done]
                if not todo:
                    continue
                data = B.build_sample(log, t0, cfg, text)
                if args.save_samples and nav == navs[0]:
                    d = Path(args.save_samples)
                    d.mkdir(parents=True, exist_ok=True)
                    torch.save({k: data[k] for k in ("image_frames", "camera_indices", "absolute_timestamps",
                                                     "ego_history_xyz")}, d / f"{sc.log_id[:8]}_{sc.t0_rel_s}_{cfg}.pt")
                prepared = to_device(prepare_model_inputs(data, model.config, model.tokenizer), device="cuda")
                lags = [max(fr["lag_ms"].values()) for fr in data["av2"]["frames"]]
                for s in todo:
                    torch.manual_seed(s)
                    gen = torch.Generator(device="cuda").manual_seed(s)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        xyz, _, _, extra = model.sample_trajectories_from_data(
                            prepared, num_traj_sets=1, num_traj_samples=1, max_generation_length=256,
                            diffusion_kwargs={"inference_step": 10, "generator": gen}, return_extra=True,
                            top_p=0.98, top_k=None, temperature=0.6)
                    p = xyz[0, 0, 0].float().cpu().numpy()
                    cot = extra.get("cot")
                    cot = cot.reshape(-1).tolist()[0] if cot is not None else None
                    a, fd = ade_fde(p, gt)
                    emit({"kind": "run", "log_id": sc.log_id, "t0_rel_s": float(sc.t0_rel_s), "motion": sc.kind,
                          "config": cfg, "nav": nav, "nav_text": text, "seed": s, "cot": cot,
                          "n_tok": int((extra["generated_ids"][0] >= 0).sum()), "ade": round(a, 3), "fde": round(fd, 3),
                          "xyz": np.round(p, 3).tolist(), "lag_ms_max": max(lags)})
                    print(f"{sc.log_id[:8]} {sc.kind:8s} {cfg:16s} {nav:8s} s{s} ADE={a:6.2f} FDE={fd:6.2f} "
                          f"| {text!r} | {cot}", flush=True)
    f.close()


if __name__ == "__main__":
    main()

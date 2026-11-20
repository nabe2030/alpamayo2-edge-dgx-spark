#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_tfcheck.py <scenes_t0.csv> <n> <out.jsonl>
# Check of the teacher forcing path (because the width of wide_eval.py's "BF16, same CoT, only the diffusion RNG differs" came out larger than the seed-to-seed width).
#   1. With the CoT of normal sampling (seed s) fixed and the diffusion RNG also set to the same s, does the trajectory match the original (is the path correct)
#   2. The width when only the diffusion RNG is changed with the same CoT (s vs s+1, 1000 vs 1001), and the seed-to-seed width (normal sampling with s vs s+1)
# Input assembly and inference settings are the same as wide_eval.py (tuned + the four optimizations, BF16, temperature 0.6, top_p 0.98, 10 diffusion steps).
import os
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a2e_fp8 as Q  # noqa: E402
import a2e_stagea as S  # noqa: E402
from qcheck import ForcedSampler  # noqa: E402
from wide_eval import av2_scene  # noqa: E402


def ade(a, b):
    return sum(math.dist(p, r) for p, r in zip(a, b)) / len(a)


def main():
    sc = pd.read_csv(sys.argv[1]).sample(int(sys.argv[2]), random_state=0)
    out = open(sys.argv[3], "w")
    S.install_processor_once()
    from alpamayo2_edge.accel import AccelConfig, enable_acceleration
    from alpamayo2_edge.checkpoint import load_model

    model, _ = load_model(str(Path(os.environ.get("A2E_MODEL", "models/Alpamayo2-Edge"))),
                          device="cuda", dtype="bfloat16", attn_implementation="sdpa", local_files_only=True)
    handle = enable_acceleration(model, AccelConfig(mode="tuned"))
    S.VisionAccel(model.vlm.model)
    vcache = S.VitFeatureCache(model.vlm.model)
    gprep = S.GpuPrep(model, cache=vcache)
    V = Q.DecodeVariants(handle)
    forced = ForcedSampler(V.dec, keep=("MaskDiscreteTrajectoryLogitsProcessor", "MaskTokenIdsLogitsProcessor"))

    def call(data, seed, noise=None, ids=None):
        prepared = gprep(data)
        torch.manual_seed(seed)
        gen = torch.Generator(device="cuda").manual_seed(noise if noise is not None else seed)
        saved = V.dec.sampler
        if ids is not None:
            forced.tokens = ids
            V.dec.sampler = forced
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                xyz, _, _, extra = model.sample_trajectories_from_data(
                    prepared, num_traj_sets=1, num_traj_samples=1, max_generation_length=256,
                    diffusion_kwargs={"inference_step": 10, "generator": gen}, return_extra=True,
                    top_p=0.98, top_k=None, temperature=0.6)
        finally:
            V.dec.sampler = saved
        return [int(x) for x in extra["generated_ids"][0].tolist() if x >= 0], xyz[0, 0, 0].float().tolist()

    for r in sc.itertuples():
        data, _, _ = av2_scene(r, "wide_ft")
        ids43, x43 = call(data, 43)
        ids44, x44 = call(data, 44)
        _, f43 = call(data, 43, noise=43, ids=ids43)        # 1. reproduce the original
        _, f44n = call(data, 43, noise=44, ids=ids43)       # 2. same CoT, only the diffusion RNG is 44
        _, f1000 = call(data, 43, noise=1000, ids=ids43)
        _, f1001 = call(data, 43, noise=1001, ids=ids43)
        rec = {"scene": r.scene_id, "reproduce_ade": ade(f43, x43), "seed_diff_ade": ade(x43, x44),
               "same_cot_43_44": ids43 == ids44, "noise_43_44_ade": ade(f43, f44n), "noise_1000_1001_ade": ade(f1000, f1001)}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(rec, flush=True)


if __name__ == "__main__":
    main()

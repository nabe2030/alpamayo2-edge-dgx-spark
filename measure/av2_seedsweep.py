#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_seedsweep.py <scenes_t0.csv> <scene_id> <seed_from> <n> <out.jsonl> [variant]
# 1 場面で seed を多く回し、判断理由の出方を数える(FP8 だけが出した判断が、BF16 でも別の seed で出るかを見るため)。
# 設定は wide_eval.py と同じ(tuned + 4 つの手、温度 0.6、top_p 0.98、拡散 10 ステップ)。variant は base / fp8all / fp8body。
import os
import json
import sys
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a2e_fp8 as Q  # noqa: E402
import a2e_stagea as S  # noqa: E402
from wide_eval import VARIANTS, av2_scene  # noqa: E402


def main():
    row = next(pd.read_csv(sys.argv[1]).query(f"scene_id == '{sys.argv[2]}'").itertuples())
    s0, n, out = int(sys.argv[3]), int(sys.argv[4]), open(sys.argv[5], "w")
    var = sys.argv[6] if len(sys.argv) > 6 else "base"
    S.install_processor_once()
    from alpamayo2_edge.accel import AccelConfig, enable_acceleration
    from alpamayo2_edge.checkpoint import load_model

    model, _ = load_model(str(Path(os.environ.get("A2E_MODEL", "models/Alpamayo2-Edge"))),
                          device="cuda", dtype="bfloat16", attn_implementation="sdpa", local_files_only=True)
    handle = enable_acceleration(model, AccelConfig(mode="tuned"))
    S.VisionAccel(model.vlm.model)
    gprep = S.GpuPrep(model, cache=S.VitFeatureCache(model.vlm.model))
    V = Q.DecodeVariants(handle)
    if var != "base":
        V.add(var, VARIANTS[var](model.vlm, handle))
    V.use(var)
    data, _, gt = av2_scene(row, "wide_ft")
    for s in range(s0, s0 + n):
        prepared = gprep(data)
        torch.manual_seed(s)
        gen = torch.Generator(device="cuda").manual_seed(s)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            xyz, _, _, extra = model.sample_trajectories_from_data(
                prepared, num_traj_sets=1, num_traj_samples=1, max_generation_length=256,
                diffusion_kwargs={"inference_step": 10, "generator": gen}, return_extra=True,
                top_p=0.98, top_k=None, temperature=0.6)
        cot = extra["cot"].reshape(-1).tolist()[0]
        out.write(json.dumps({"scene": row.scene_id, "variant": var, "seed": s, "cot": cot,
                              "xyz": xyz[0, 0, 0].float().tolist()}, ensure_ascii=False) + "\n")
        out.flush()
        print(s, cot, flush=True)


if __name__ == "__main__":
    main()

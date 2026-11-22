#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_firsttoken.py <plan.csv> <stopgo.jsonl> <out.jsonl> <variants>
# Distribution of the first reasoning token in the stop/go scenes, for several decode variants
# (e.g. fp8body, fp4rtn, fp4gptq_d10, fp4gptq_d10_h8). Summarized by summarize_firsttoken.py.
#   Same way of taking it as the _tf.jsonl of av2_stopgo.py: the BF16 reasoning of the first recorded seed of each scene
#   (gen_ids of base in stopgo.jsonl) is teacher-forced, and the distribution at position 0 is stored with temperature
#   0.6, top 30. base is taken again and compared in the same process.
#   plan.csv and stopgo.jsonl come from av2_stopgo.py (run on the 139 stop/go scenes). The scene list is
#   $A2E_SCENES (default raw/wide/scenes_t0.csv under the current directory, i.e. $A2E_WORK after `source env.sh`);
#   the model is $A2E_MODEL. FP4 variants (names starting with fp4) come from a2e_fp4q.py (and a2e_fp4.py).
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a2e_fp8 as Q  # noqa: E402
import a2e_stagea as S  # noqa: E402
from qcheck import ForcedSampler  # noqa: E402
from wide_eval import VARIANTS, av2_scene  # noqa: E402


def main():
    plan = pd.read_csv(sys.argv[1])
    first = {}
    for l in open(sys.argv[2]):
        r = json.loads(l)
        first.setdefault(r["scene"], (r["seed"], r["out"]["base"]["gen_ids"]))
    out = open(sys.argv[3], "w")
    variants = sys.argv[4].split(",")
    scenes = pd.read_csv(os.environ.get("A2E_SCENES", "raw/wide/scenes_t0.csv")).set_index("scene_id")
    S.install_processor_once()
    from alpamayo2_edge.accel import AccelConfig, enable_acceleration
    from alpamayo2_edge.checkpoint import load_model

    model, _ = load_model(os.environ["A2E_MODEL"],
                          device="cuda", dtype="bfloat16", attn_implementation="sdpa", local_files_only=True)
    handle = enable_acceleration(model, AccelConfig(mode="tuned"))
    S.VisionAccel(model.vlm.model)
    gprep = S.GpuPrep(model, cache=S.VitFeatureCache(model.vlm.model))
    V = Q.DecodeVariants(handle)
    if any(v.startswith("fp4") for v in variants):  # FP4 variants
        import a2e_fp4
        import a2e_fp4q
        VARIANTS.update(a2e_fp4q.VARIANTS)
        VARIANTS.update(a2e_fp4.VARIANTS)
    for v in variants:
        if v != "base":
            V.add(v, VARIANTS[v](model.vlm, handle))
    forced = ForcedSampler(V.dec, keep=("MaskDiscreteTrajectoryLogitsProcessor", "MaskTokenIdsLogitsProcessor"))
    tok = model.tokenizer
    for p in plan.itertuples():
        if p.scene_id not in first:
            continue
        seed, ids = first[p.scene_id]
        row = pd.Series({**scenes.loc[p.scene_id].to_dict(), "scene_id": p.scene_id})
        data, _, _ = av2_scene(next(pd.DataFrame([row]).itertuples()), "wide_ft")
        rec = {"scene": p.scene_id, "tier": p.tier, "ref_seed": seed, "top": {}}
        for v in variants:
            V.use(v)
            prepared = gprep(data)
            torch.manual_seed(0)
            gen = torch.Generator(device="cuda").manual_seed(0)
            saved = V.dec.sampler
            forced.tokens = ids
            V.dec.sampler = forced
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model.sample_trajectories_from_data(
                        prepared, num_traj_sets=1, num_traj_samples=1, max_generation_length=256,
                        diffusion_kwargs={"inference_step": 10, "generator": gen}, return_extra=True,
                        top_p=0.98, top_k=None, temperature=0.6)
            finally:
                V.dec.sampler = saved
            pr = torch.softmax(forced.scores[0].float() / 0.6, dim=-1)
            top = torch.topk(pr, 30)
            rec["top"][v] = [[tok.convert_ids_to_tokens(int(i)), int(i), round(float(q), 6)] for q, i in zip(top.values, top.indices)]
        V.use("base")
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        print(time.strftime("%H:%M:%S"), p.scene_id, flush=True)


if __name__ == "__main__":
    main()

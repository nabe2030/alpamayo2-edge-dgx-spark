#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_stopgo.py <plan.csv> <out.jsonl>
# 止まるべき場面・譲る判断が出る場面で seed を増やし、BF16 と FP8(本体だけ/本体 + 出力層)の
#   「進む」の割合を比べるための生の記録を取る。plan.csv の列: scene_id, tier, seed_from, n_seeds。
#   設定は wide_eval.py / av2_seedsweep.py と同じ(tuned + 4 つの手、温度 0.6、top_p 0.98、拡散 10 ステップ、
#   torch.manual_seed(s) と拡散の乱数 s)。場面ごとに描き直しは 1 回。seed ごとに 3 つの変種を続けて回す(同じ seed で対にする)。
#   途中から再開できる(out にある (scene, seed) は飛ばす)。84℃ で休む。
#   場面ごとに 1 回、1 トークン目の分布も記録する(<out>_tf.jsonl): BF16 の最初の seed の判断理由を teacher forcing で流し
#   (qcheck.ForcedSampler、乱数を使わない)、位置 0 の分布(温度 0.6 をかけた softmax)の上位 30 を変種ごとに残す。
#   「どのトークンで分かれたか」が 1 トークン目に集まっていたため(5 seed の記録の試しの集計)。
import os
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a2e_fp8 as Q  # noqa: E402
import a2e_stagea as S  # noqa: E402
from a2e_bench import thermal_gate  # noqa: E402
from qcheck import ForcedSampler  # noqa: E402
from wide_eval import VARIANTS, av2_scene  # noqa: E402

VARS = ("base", "fp8body", "fp8all")


def main():
    plan = pd.read_csv(sys.argv[1])
    outp = Path(sys.argv[2])
    done = set()
    if outp.exists():
        for l in outp.read_text().splitlines():
            r = json.loads(l)
            done.add((r["scene"], r["seed"]))
    out = open(outp, "a")
    tfp = outp.with_name(outp.stem + "_tf.jsonl")
    tf_done = {json.loads(l)["scene"] for l in tfp.read_text().splitlines()} if tfp.exists() else set()
    tfo = open(tfp, "a")
    scenes = pd.read_csv(HERE.parent / "raw/wide/scenes_t0.csv").set_index("scene_id")
    S.install_processor_once()
    from alpamayo2_edge.accel import AccelConfig, enable_acceleration
    from alpamayo2_edge.checkpoint import load_model

    model, _ = load_model(str(Path(os.environ.get("A2E_MODEL", "models/Alpamayo2-Edge"))),
                          device="cuda", dtype="bfloat16", attn_implementation="sdpa", local_files_only=True)
    handle = enable_acceleration(model, AccelConfig(mode="tuned"))
    S.VisionAccel(model.vlm.model)
    gprep = S.GpuPrep(model, cache=S.VitFeatureCache(model.vlm.model))
    V = Q.DecodeVariants(handle)
    for v in VARS[1:]:
        V.add(v, VARIANTS[v](model.vlm, handle))
    forced = ForcedSampler(V.dec, keep=("MaskDiscreteTrajectoryLogitsProcessor", "MaskTokenIdsLogitsProcessor"))
    tok = model.tokenizer

    def first_token_dist(data, ids):
        rec = {}
        for v in VARS:
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
            rec[v] = [[tok.convert_ids_to_tokens(int(i)), int(i), round(float(p_), 6)] for p_, i in zip(top.values, top.indices)]
        V.use("base")
        return rec

    def log(m):
        print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)

    def past_7am():
        import datetime
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
        return 7 <= now.hour < 12

    for p in plan.itertuples():
        if past_7am():  # 07:00 JST を過ぎたら新しい場面を始めない(指示 §4)
            log("07:00 JST を過ぎたので止める")
            break
        seeds = [s for s in range(p.seed_from, p.seed_from + p.n_seeds) if (p.scene_id, s) not in done]
        if not seeds:
            continue
        row = scenes.loc[p.scene_id]
        row = pd.Series({**row.to_dict(), "scene_id": p.scene_id})
        data, _, _ = av2_scene(next(pd.DataFrame([row]).itertuples()), "wide_ft")
        t = time.perf_counter()
        for s in seeds:
            thermal_gate(log)
            rec = {"scene": p.scene_id, "tier": p.tier, "seed": s, "out": {}}
            for v in VARS:
                V.use(v)
                prepared = gprep(data)
                torch.manual_seed(s)
                gen = torch.Generator(device="cuda").manual_seed(s)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    xyz, _, _, extra = model.sample_trajectories_from_data(
                        prepared, num_traj_sets=1, num_traj_samples=1, max_generation_length=256,
                        diffusion_kwargs={"inference_step": 10, "generator": gen}, return_extra=True,
                        top_p=0.98, top_k=None, temperature=0.6)
                rec["out"][v] = {"cot": extra["cot"].reshape(-1).tolist()[0],
                                 "gen_ids": [int(x) for x in extra["generated_ids"][0].tolist() if x >= 0],
                                 "xyz": [[round(c, 4) for c in q] for q in xyz[0, 0, 0].float().tolist()]}
            V.use("base")
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
        if p.scene_id not in tf_done:
            first = next(json.loads(l) for l in open(outp) if json.loads(l)["scene"] == p.scene_id)
            tfo.write(json.dumps({"scene": p.scene_id, "tier": p.tier, "ref_seed": first["seed"],
                                  "top": first_token_dist(data, first["out"]["base"]["gen_ids"])}, ensure_ascii=False) + "\n")
            tfo.flush()
            tf_done.add(p.scene_id)
        log(f"{p.scene_id} tier {p.tier} seeds {len(seeds)} {time.perf_counter() - t:.0f}s "
            f"last: {rec['out']['base']['cot'][:50]!r} / {rec['out']['fp8all']['cot'][:50]!r}")
    log("done")


if __name__ == "__main__":
    main()

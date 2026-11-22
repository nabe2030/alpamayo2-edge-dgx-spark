#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
"""PyTorch (BF16, plain HF path) references for several AV2 scenes, in the layout trt_run.py reads:
per scene <out>/<scene_id>/{prompt.txt, collapsed_ids.npy, img_XX.png, ref.json}.
ref.json holds the greedy CoT ids (Alpamayo's logit masks applied) and, for every greedy step,
the top-20 log-probabilities, so TensorRT outputs can be compared step by step.
Run in the Alpamayo2-Edge venv; scenes come from a scene csv (wide_eval format).
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--root", default=".", help="base directory of the data_dir column of the scene csv (default: current directory, i.e. $A2E_WORK after `source env.sh`)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-gen", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    # av2_build / av2_nav come from av2_input/ of this repository (PYTHONPATH set by env.sh)
    import av2_build as B
    import av2_nav as N
    from torchvision.transforms.functional import InterpolationMode, resize
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    from alpamayo2_edge.checkpoint import load_model
    from alpamayo2_edge.helper import prepare_model_inputs, to_device, _canonical_inputs, get_processor
    from alpamayo2_edge.chat_template import build_conversation
    from alpamayo2_edge.models.utils import fuse_traj_tokens

    model, _ = load_model(a.model, device="cuda", dtype="bfloat16", attn_implementation="sdpa", local_files_only=True)
    cfg, tok = model.config, model.tokenizer
    ip = get_processor(tok, cfg).image_processor
    tid = cfg.traj_ids
    masked = list(range(tid["history_id0"], tid["history_id0"] + cfg.history_vocab_size)) + \
        list(range(tid["future_id0"], tid["future_id0"] + cfg.future_vocab_size))
    eos = model.vlm.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    blocked = [t for t in eos if t != tid["future_start"]]
    img_tok = cfg.vlm_config.image_token_id
    rows = list(csv.DictReader(open(a.scenes)))[: a.n]
    for r in rows:
        sid = r["scene_id"]
        od = Path(a.out) / sid
        od.mkdir(parents=True, exist_ok=True)
        # Same sample construction as measure/wide_eval.py av2_scene (absolute t0_ns).
        lg = B.AV2Log(str(Path(a.root) / r["data_dir"]))
        t0_ns = int(r["t0_ns"])
        data = B.build_sample(lg, t0_ns, B.CONFIGS["wide_ft"], N.nav_from_future(lg, t0_ns)["text"])
        data.pop("av2", None)
        prep = to_device(prepare_model_inputs(data, cfg, tok), device="cuda")
        td = dict(prep["tokenized_data"])
        ids = fuse_traj_tokens(model.history_traj_tokenizer, model.future_traj_tokenizer, td["input_ids"],
                               {k: prep[k] for k in ("ego_history_xyz", "ego_history_rot")}, tid)
        ids_np = ids[0].cpu().numpy()
        col = [int(ids_np[0])] + [int(t) for p, t in zip(ids_np[:-1], ids_np[1:]) if not (t == img_tok and p == img_tok)]
        np.save(od / "collapsed_ids.npy", np.array(col, np.int32))
        (od / "prompt.txt").write_text(tok.decode(col, skip_special_tokens=False))
        grid = td["image_grid_thw"].cpu().numpy()
        msgs = build_conversation(_canonical_inputs(data, cfg), num_tokens_per_history_traj=cfg.tokens_per_history_traj,
                                  include_camera_ids=cfg.include_camera_ids, system_prompt=getattr(cfg, "system_prompt", None))
        n = 0
        for m in msgs:
            for e in m["content"]:
                if e["type"] != "image":
                    continue
                fr = e["image"].squeeze(0)
                h, w = smart_resize(*fr.shape[-2:], factor=ip.patch_size * ip.merge_size,
                                    min_pixels=e["min_pixels"], max_pixels=e["max_pixels"])
                im = resize(fr, [h, w], interpolation=InterpolationMode.BICUBIC)
                assert (h // ip.patch_size, w // ip.patch_size) == tuple(grid[n, 1:])
                Image.fromarray(im.permute(1, 2, 0).cpu().numpy()).save(od / f"img_{n:02d}.png")
                n += 1
        seq, tops = [], []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = model.vlm(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=td["pixel_values"],
                          image_grid_thw=td["image_grid_thw"], mm_token_type_ids=td["mm_token_type_ids"], use_cache=True)
            lg = o.logits[0, -1].float()
            cache = o.past_key_values
            for step in range(a.max_gen):
                lg[masked] = -float("inf"); lg[blocked] = -float("inf")
                lp = torch.log_softmax(lg, -1)
                t = torch.topk(lp, 20)
                tops.append([t.indices.tolist(), [round(v, 5) for v in t.values.tolist()]])
                nxt = int(t.indices[0])
                seq.append(nxt)
                if nxt == tid["future_start"]:
                    break
                o = model.vlm(input_ids=torch.tensor([[nxt]], device="cuda"), past_key_values=cache, use_cache=True)
                cache = o.past_key_values
                lg = o.logits[0, -1].float()
        ref = {"scene_id": sid, "prompt_len": int(ids.shape[1]), "collapsed_len": len(col), "n_images": n,
               "masked_ranges": [[tid["history_id0"], cfg.history_vocab_size], [tid["future_id0"], cfg.future_vocab_size]],
               "blocked_eos": blocked, "anchor": tid["future_start"], "greedy_ids": seq, "greedy_text": tok.decode(seq),
               "greedy_top20": tops}
        json.dump(ref, open(od / "ref.json", "w"), ensure_ascii=False)
        print(sid, ids.shape[1], len(seq), repr(ref["greedy_text"][:80]), flush=True)


if __name__ == "__main__":
    main()

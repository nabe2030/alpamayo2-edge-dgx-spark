#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
"""Dump a PyTorch reference for one AV2 sample, to feed the same prompt to TensorRT Edge-LLM.

Writes <out>/:
  input_ids.npy          full prompt ids (trajectory tokens fused), int32 [L]
  collapsed_ids.npy      same, each run of image-pad tokens collapsed to one pad (Edge-LLM expands it)
  image_grid_thw.npy     [n_img, 3]
  img_XX.png             uint8 RGB images after the checkpoint's own smart_resize (bicubic, on uint8)
  positions_hf.npy       HF get_rope_index positions [3, L-1] and rope delta
  ref.json               greedy CoT ids / first-step logits top-k / official sampled CoT (seed)
  logits0.npy            first generated-token logits (masked as in Alpamayo), float32
Run with the Alpamayo2-Edge venv (PYTHONPATH = upstream src).
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--greedy-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=43)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    from torchvision.transforms.functional import InterpolationMode, resize
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    from transformers.generation.logits_process import LogitsProcessorList
    from alpamayo2_edge.checkpoint import load_model
    from alpamayo2_edge.helper import prepare_model_inputs, to_device, _canonical_inputs, get_processor
    from alpamayo2_edge.chat_template import build_conversation
    from alpamayo2_edge.models.utils import fuse_traj_tokens
    from alpamayo2_edge.models import alpamayo2_edge as AM

    data = torch.load(a.input, weights_only=False)
    model, _ = load_model(a.model, device="cuda", dtype="bfloat16", attn_implementation="sdpa",
                          local_files_only=True)
    cfg = model.config
    prepared = prepare_model_inputs(data, cfg, model.tokenizer)
    prepared = to_device(prepared, device="cuda")
    td = dict(prepared["tokenized_data"])
    ids = fuse_traj_tokens(model.history_traj_tokenizer, model.future_traj_tokenizer, td["input_ids"],
                           {k: prepared[k] for k in ("ego_history_xyz", "ego_history_rot")}, cfg.traj_ids)
    td["input_ids"] = ids
    L = ids.shape[1]
    img_tok = cfg.vlm_config.image_token_id
    ids_np = ids[0].cpu().numpy().astype(np.int32)
    np.save(out / "input_ids.npy", ids_np)
    col = [int(ids_np[0])] + [int(t) for p, t in zip(ids_np[:-1], ids_np[1:]) if not (t == img_tok and p == img_tok)]
    np.save(out / "collapsed_ids.npy", np.array(col, np.int32))
    grid = td["image_grid_thw"].cpu().numpy()
    np.save(out / "image_grid_thw.npy", grid)

    # Re-create the resized uint8 images exactly as helper._tokenize_messages does.
    processor = get_processor(model.tokenizer, cfg)
    ip = processor.image_processor
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
            assert (h // ip.patch_size, w // ip.patch_size) == tuple(grid[n, 1:]), (h, w, grid[n])
            Image.fromarray(im.permute(1, 2, 0).cpu().numpy()).save(out / f"img_{n:02d}.png")
            n += 1
    assert n == grid.shape[0]

    vm = model.vlm.model
    tt = td["mm_token_type_ids"]
    pos, delta = vm.get_rope_index(ids[:, :-1].cpu(), tt[:, :-1].cpu(), td["image_grid_thw"].cpu(), None,
                                   attention_mask=torch.ones_like(ids[:, :-1]).cpu())
    np.save(out / "positions_hf.npy", pos[:, 0].numpy())

    # Masks used by Alpamayo during CoT generation.
    tid = cfg.traj_ids
    masked = list(range(tid["history_id0"], tid["history_id0"] + cfg.history_vocab_size)) + \
        list(range(tid["future_id0"], tid["future_id0"] + cfg.future_vocab_size))
    gc = model.vlm.generation_config
    eos = gc.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    blocked_eos = [t for t in eos if t != tid["future_start"]]

    # Greedy decode with the same masks (plain HF, no accel).
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        o = model.vlm(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=td["pixel_values"],
                      image_grid_thw=td["image_grid_thw"], mm_token_type_ids=tt, use_cache=True)
        lg = o.logits[0, -1].float()
        lg[masked] = -float("inf"); lg[blocked_eos] = -float("inf")
        np.save(out / "logits0.npy", lg.cpu().numpy())
        top = torch.topk(lg, 20)
        cache = o.past_key_values
        seq = []
        nxt = int(top.indices[0])
        steps = 0
        while steps < a.greedy_tokens:
            seq.append(nxt); steps += 1
            if nxt == tid["future_start"]:
                break
            o = model.vlm(input_ids=torch.tensor([[nxt]], device="cuda"), past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            lg2 = o.logits[0, -1].float(); lg2[masked] = -float("inf"); lg2[blocked_eos] = -float("inf")
            nxt = int(lg2.argmax())
    tok = model.tokenizer
    ref = {"prompt_len": L, "collapsed_len": len(col), "n_images": int(n), "grid": grid.tolist(),
           "rope_delta": int(delta.reshape(-1)[0]), "image_token_id": img_tok,
           "masked_ranges": [[tid["history_id0"], cfg.history_vocab_size], [tid["future_id0"], cfg.future_vocab_size]],
           "blocked_eos": blocked_eos, "anchor": tid["future_start"],
           "greedy_ids": seq, "greedy_text": tok.decode(seq),
           "top20_ids": top.indices.tolist(), "top20_logits": top.values.tolist()}

    # Official sampled CoT (seed) through the full model, for reference.
    torch.manual_seed(a.seed)
    gen = torch.Generator(device="cuda").manual_seed(a.seed)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, _, _, extra = model.sample_trajectories_from_data(
            prepared, num_traj_sets=1, num_traj_samples=1, max_generation_length=256,
            diffusion_kwargs={"inference_step": 10, "generator": gen}, return_extra=True,
            top_p=0.98, top_k=None, temperature=0.6)
    g = extra["generated_ids"][0].tolist()
    ref["sampled_seed"] = a.seed
    ref["sampled_ids"] = [int(x) for x in g if x >= 0]
    cot = extra.get("cot")
    ref["sampled_cot"] = cot.reshape(-1).tolist()[0] if cot is not None else None
    json.dump(ref, open(out / "ref.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: ref[k] for k in ("prompt_len", "collapsed_len", "n_images", "rope_delta", "greedy_text", "sampled_cot")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
"""Run the Alpamayo2-Edge reasoner (VLM part) on TensorRT Edge-LLM with the exact PyTorch prompt.

Inputs come from trt_ref_dump.py: collapsed_ids.npy (one image pad per image; the runtime expands it),
img_XX.png (already resized by the checkpoint's own smart_resize, so do_resize=False).
Alpamayo's CoT logit masks (trajectory vocab, text EOS) are applied as logit_bias.
"""
import argparse, glob, json, os, statistics, time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--mode", choices=["greedy", "sample"], default="greedy")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--max-gen", type=int, default=64)
    ap.add_argument("--profile", type=int, default=0)
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--perturb", type=int, default=0,
                    help="1: change one corner pixel per run so the content-addressed encoder (ViT) cache misses")
    ap.add_argument("--stream-metrics", type=int, default=0,
                    help="1: attach a StreamChannel (note: with it the stop string no longer ends generation)")
    ap.add_argument("--out", default="")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    from tensorrt_edgellm import runtime
    rt = runtime.load()
    ref = json.load(open(os.path.join(a.ref, "ref.json")))
    ids = np.load(os.path.join(a.ref, "collapsed_ids.npy")).astype(int).tolist()
    # The Cosmos3-Edge vision runner (a Qwen3-VL runner) re-tokenizes the formatted prompt string and
    # rejects pre-tokenized ids, so the prompt goes in as text: the exact decode of the PyTorch ids
    # (round-trip checked to give the same ids; prompt.txt comes from trt_ref_dump.py / trt_ref_multi.py).
    prompt = open(os.path.join(a.ref, "prompt.txt")).read()
    pngs = sorted(glob.glob(os.path.join(a.ref, "img_*.png")))
    bias = {}
    # Edge-LLM caps logit_bias at 1024 entries per request, so the 4000 trajectory-vocab ids cannot be
    # masked; only the text EOS is blocked. --no-mask drops even that.
    if not a.no_mask:
        for t in ref["blocked_eos"]:
            bias[t] = -100.0
    t = time.perf_counter()
    llm = rt.LLMRuntime(engine_dir=a.engine_dir, multimodal_engine_dir=a.engine_dir, checkpoint_dir=a.model_dir)
    load_s = time.perf_counter() - t
    if a.profile:
        rt.set_profiling_enabled(True)

    import io
    from PIL import Image
    base_imgs = [np.array(Image.open(p).convert("RGB")) for p in pngs]

    def make_request(seed, run_idx):
        imgs = []
        for arr in base_imgs:
            if a.perturb:
                # Edge-LLM caches ViT outputs by image content (256 MiB, not configurable from Python).
                # Touch one corner pixel per run so every run re-encodes, as in the PyTorch single-shot runs.
                arr = arr.copy()
                arr[0, 0, 0] = (int(arr[0, 0, 0]) + 1 + run_idx) % 256
                buf = io.BytesIO()
                Image.fromarray(arr).save(buf, format="PNG", compress_level=1)
                im = rt.load_image_from_bytes(buf.getvalue())
            else:
                buf = io.BytesIO()
                Image.fromarray(arr).save(buf, format="PNG", compress_level=1)
                im = rt.load_image_from_bytes(buf.getvalue())
            im.do_resize = False
            imgs.append(im)
        req = rt.Request(messages=[rt.create_text_message("user", prompt)])
        req.image_buffers = imgs
        req.stop_strings = ["<|traj_future_start|>"]
        req.logit_bias = bias
        req.sampling_seed = seed
        r = rt.LLMGenerationRequest()
        r.requests = [req]
        r.apply_chat_template = False
        r.add_generation_prompt = False
        r.skip_special_tokens = False
        r.max_generate_length = a.max_gen
        r.sampling_seed = seed
        if a.stream_metrics:
            ch = rt.StreamChannel.create()
            r.stream_channels = [ch]
        if a.mode == "greedy":
            r.temperature = 0.0
            r.top_k = 1
            r.top_p = 1.0
        else:
            r.temperature = 0.6
            r.top_p = 0.98
            r.top_k = 0
        return r, (ch if a.stream_metrics else None)

    recs = []
    for i in range(a.warmup + a.runs):
        seed = a.seed + (i - a.warmup if a.mode == "sample" else 0)
        req, ch = make_request(seed, i)  # image encode/decode (PNG) is outside the timed span
        t0 = time.perf_counter()
        resp = llm.handle_request(req)
        ms = (time.perf_counter() - t0) * 1000
        out_ids = list(resp.output_ids[0])
        sm = None
        if ch is not None:
            last = None
            while True:
                c = ch.try_pop()
                if c is None:
                    break
                last = c
            if last is not None and last.metrics is not None:
                m = last.metrics
                sm = {k: getattr(m, k) for k in ("prompt_seconds", "generation_seconds", "processed_prompt_tokens",
                                                   "generated_tokens", "decoded_tokens", "prefill_start_ns")}
        rec = {"idx": i, "warm": i >= a.warmup, "ms": ms, "n_out": len(out_ids), "out_ids": out_ids,
               "text": resp.output_texts[0], "prompt_tokens": list(resp.prompt_token_counts),
               "finish": str(resp.finish_reasons[0]), "seed": seed, "stream_metrics": sm}
        recs.append(rec)
        print(json.dumps({k: rec[k] for k in ("idx", "ms", "n_out", "text", "prompt_tokens", "finish", "stream_metrics")}, ensure_ascii=False), flush=True)
    warm = [r["ms"] for r in recs if r["warm"]]
    summ = {"label": a.label, "mode": a.mode, "load_s": load_s, "runs": len(warm),
            "median_ms": statistics.median(warm) if warm else None, "max_ms": max(warm) if warm else None,
            "ref_greedy_ids": ref["greedy_ids"]}
    g = recs[-1]["out_ids"]
    if a.mode == "greedy":
        rg = ref["greedy_ids"]
        same = 0
        for x, y in zip(g, rg):
            if x != y:
                break
            same += 1
        summ.update({"greedy_prefix_match": same, "trt_len": len(g), "ref_len": len(rg), "identical": g == rg})
    try:
        pm, gm, mm = llm.get_prefill_metrics(), llm.get_generation_metrics(), llm.get_multimodal_metrics()
        summ["metrics"] = {"prefill_runs": pm.get_total_runs(), "computed_tokens": pm.computed_tokens,
                           "gen_runs": gm.get_total_runs(), "generated_tokens": gm.generated_tokens,
                           "mm_runs": mm.get_total_runs(), "image_tokens": mm.total_image_tokens, "images": mm.total_images}
    except Exception as e:  # metrics are best-effort
        summ["metrics_error"] = repr(e)
    print(json.dumps(summ, ensure_ascii=False))
    if a.out:
        with open(a.out, "a") as f:
            for r in recs:
                f.write(json.dumps({"kind": "run", "label": a.label, **r}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"kind": "summary", **summ}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

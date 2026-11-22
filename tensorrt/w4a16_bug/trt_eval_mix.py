#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
"""Same as ../trt_eval_multi.py, plus --mm-engine-dir to mix LLM / visual engines and --limit.
Compare TensorRT Edge-LLM greedy CoT against the PyTorch references from trt_ref_multi.py.

Per scene: greedy CoT identical?, length of the common prefix, and per step (while both agree) the
divergence between TRT and PyTorch next-token distributions over PyTorch's top-20 (KL restricted to
those 20 ids, renormalized), plus top-1 agreement at step 0.
"""
import argparse, glob, io, json, math, os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--mm-engine-dir", default=None, help="visual engine dir (default: same as --engine-dir)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N scenes")
    a = ap.parse_args()
    from PIL import Image
    from tensorrt_edgellm import runtime
    rt = runtime.load()
    llm = rt.LLMRuntime(engine_dir=a.engine_dir, multimodal_engine_dir=a.mm_engine_dir or a.engine_dir, checkpoint_dir=a.model_dir)
    res = []
    refs = sorted(glob.glob(os.path.join(a.refs, "*", "ref.json")))
    for d in (refs[:a.limit] if a.limit else refs):
        sd = os.path.dirname(d)
        ref = json.load(open(d))
        prompt = open(os.path.join(sd, "prompt.txt")).read()
        imgs = []
        for p in sorted(glob.glob(os.path.join(sd, "img_*.png"))):
            buf = io.BytesIO(); Image.open(p).save(buf, format="PNG", compress_level=1)
            im = rt.load_image_from_bytes(buf.getvalue()); im.do_resize = False; imgs.append(im)
        req = rt.Request(messages=[rt.create_text_message("user", prompt)])
        req.image_buffers = imgs
        req.stop_strings = ["<|traj_future_start|>"]
        req.logit_bias = {t: -100.0 for t in ref["blocked_eos"]}
        r = rt.LLMGenerationRequest()
        r.requests = [req]
        r.apply_chat_template = False; r.add_generation_prompt = False; r.skip_special_tokens = False
        r.max_generate_length = 64; r.temperature = 0.0; r.top_k = 1; r.top_p = 1.0
        r.num_logprobs = a.k
        resp = llm.handle_request(r)
        out = list(resp.output_ids[0])
        g = ref["greedy_ids"]
        pre = 0
        for x, y in zip(out, g):
            if x != y:
                break
            pre += 1
        kls = []
        lps = resp.logprobs[0] if len(resp.logprobs) else []
        for s in range(min(pre + 1, len(lps), len(ref["greedy_top20"]))):
            ids, vals = ref["greedy_top20"][s]
            trt = {e.token_id: e.logprob for e in lps[s]}
            # PyTorch distribution over its top-20 vs TRT's over the same ids (missing -> floor)
            floor = min(trt.values()) - 2.0 if trt else -30.0
            p = np.exp(np.array(vals)); p /= p.sum()
            q = np.exp(np.array([trt.get(i, floor) for i in ids])); q /= q.sum()
            kls.append(float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)))))
        top1_0 = (lps[0][0].token_id if lps and len(lps[0]) else None)
        rec = {"label": a.label, "scene": ref["scene_id"], "identical": out == g, "prefix": pre, "ref_len": len(g),
               "trt_len": len(out), "prompt_tokens": list(resp.prompt_token_counts), "ref_prompt": ref["prompt_len"],
               "top1_step0_same": top1_0 == g[0] if g else None, "kl_steps": kls,
               "trt_text": resp.output_texts[0], "ref_text": ref["greedy_text"]}
        res.append(rec)
        print(json.dumps({k: rec[k] for k in ("scene", "identical", "prefix", "ref_len", "trt_len", "prompt_tokens", "ref_prompt")}),
              "kl0=%.2e" % kls[0] if kls else "", flush=True)
    with open(a.out, "w") as f:
        for r_ in res:
            f.write(json.dumps(r_, ensure_ascii=False) + "\n")
    n = len(res)
    allkl = [k for r_ in res for k in r_["kl_steps"]]
    print(json.dumps({"label": a.label, "scenes": n, "identical": sum(r_["identical"] for r_ in res),
                      "top1_step0_same": sum(bool(r_["top1_step0_same"]) for r_ in res),
                      "kl_median": float(np.median(allkl)) if allkl else None,
                      "kl_mean": float(np.mean(allkl)) if allkl else None, "kl_steps": len(allkl)}))


if __name__ == "__main__":
    main()

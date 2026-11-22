#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
"""Run one request on an Edge-LLM engine and print the greedy output and top logprobs.

With EDGELLM_DUMP_LOGITS_KVCACHE_LAYERS / EDGELLM_DUMP_LOGITS_KVCACHE_DIR set in the environment, the
C++ runtime's LayerDebugger writes per-round logits and per-layer KV cache to a safetensors file.
--scene <dir> uses an AV2 reference scene from trt_ref_multi.py (prompt + images); otherwise --text.
"""
import argparse, glob, io, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--mm-engine-dir", default=None)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--text", default="The capital of France is")
    ap.add_argument("--scene", default=None)
    ap.add_argument("--max-new", type=int, default=8)
    a = ap.parse_args()
    from PIL import Image
    from tensorrt_edgellm import runtime
    rt = runtime.load()
    llm = rt.LLMRuntime(engine_dir=a.engine_dir, multimodal_engine_dir=a.mm_engine_dir or a.engine_dir,
                        checkpoint_dir=a.model_dir)
    imgs = []
    if a.scene:
        prompt = open(os.path.join(a.scene, "prompt.txt")).read()
        for p in sorted(glob.glob(os.path.join(a.scene, "img_*.png"))):
            buf = io.BytesIO(); Image.open(p).save(buf, format="PNG", compress_level=1)
            im = rt.load_image_from_bytes(buf.getvalue()); im.do_resize = False; imgs.append(im)
    else:
        prompt = a.text
    req = rt.Request(messages=[rt.create_text_message("user", prompt)])
    if imgs:
        req.image_buffers = imgs
    r = rt.LLMGenerationRequest()
    r.requests = [req]
    r.apply_chat_template = False; r.add_generation_prompt = False; r.skip_special_tokens = False
    r.max_generate_length = a.max_new; r.temperature = 0.0; r.top_k = 1; r.top_p = 1.0
    r.num_logprobs = 5
    resp = llm.handle_request(r)
    lps = resp.logprobs[0] if len(resp.logprobs) else []
    print(json.dumps({"engine": a.engine_dir, "prompt_tokens": list(resp.prompt_token_counts),
                      "out_ids": list(resp.output_ids[0]), "text": resp.output_texts[0],
                      "top5_step0": [(e.token_id, round(e.logprob, 4)) for e in lps[0]] if lps else None},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

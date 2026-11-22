#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# a2e_bench2.py — measurement. With the same stage boundaries and the same clock as a2e_bench.py,
# stacks up the four optimizations (a2e_stagea.py) with --feat and measures them.
#
#   --feat   empty = as official / p = #1 load the processor once (PR #11) / g = #9 resize images on the GPU
#            v = #3 compile the ViT too + CUDA graphs / c = #2 reuse ViT results per image
#   --mode single : single-run measurement. Clears the ViT cache every time and feeds the same sample
#   --mode stream : continuous measurement. Feeds, in order, samples whose t0 advances by --stride within the same clip.
#                   Clears the cache at the start of each pass (the first step of a pass is counted separately as the "cold" state).
#
#   --q      quantized CoT decoding: body / head / all = FP8 (a2e_fp8); fp4 / fp4_h8 / fp4_h4 = FP4 body (a2e_fp4q) with the
#            output layer in BF16 / FP8 / FP4. FP4 example: A2E_FP4_METHOD=gptq:0.1 A2E_CALIB_IN=<H file> ... --q fp4
#
# Recorded per call: e2e (including preprocessing, wall clock synchronized before and after), per stage (CUDA events synchronized at boundaries),
# CoT token sequence and text, trajectory (xyz of candidate 0, 64 points), number of images passed through the ViT.
import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a2e_bench import Marks, gpu_temp, thermal_gate, versions, wrap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--feat", default="", help="comma list of p,g,v,c")
    ap.add_argument("--accel", choices=["off", "compile", "tuned"], default="tuned")
    ap.add_argument("--mode", choices=["single", "stream"], default="single")
    ap.add_argument("--input", default="", help="single: pt:<path>")
    ap.add_argument("--clip-dir", default="")
    # AV2: the source of continuous input is an AV2 log (av2_stream.AV2StreamSource). t0 is in µs from the start of the log
    ap.add_argument("--av2-log", default="")
    ap.add_argument("--av2-config", default="wide_ft")
    ap.add_argument("--av2-nav", default="gt")
    # inline = do the re-rendering (from JPEG loading) inside each step, and also record the time combined with inference
    ap.add_argument("--av2-render", choices=["pre", "inline"], default="pre")
    ap.add_argument("--av2-decode", default="pil_threads", help="how inline reads JPEGs (av2_render_fast)")
    ap.add_argument("--av2-out", default="pinned", help="how inline holds the rendered images (out_mode of av2_render_fast: cpu / pinned)")
    ap.add_argument("--t0-start", type=int, default=2_000_000, help="stream: starting t0 (lower bound of the official loader: 2.0 s)")
    ap.add_argument("--measure-from", type=int, default=4_000_000,
                    help="stream: count toward measurement from this t0 (before that is the period that fills the cache)")
    ap.add_argument("--t0-stop", type=int, default=19_900_000)
    ap.add_argument("--prep-steps", type=int, default=30, help="stream: number of steps in the warm-up pass")
    ap.add_argument("--prep-steps-next", type=int, default=5, help="stream: number of steps in the warm-up pass for the 2nd and later logs")
    ap.add_argument("--stride", type=int, default=100_000)
    ap.add_argument("--passes", type=int, default=2, help="stream: number of measured passes (plus one warm-up pass)")
    ap.add_argument("--nav-text", default="Turn right in 30m")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--seed-mode", choices=["fixed", "vary"], default="fixed")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--sync-stages", type=int, default=1)
    ap.add_argument("--vit-mode", default="default", help="inductor mode for #3")
    ap.add_argument("--pin", type=int, default=0, help="#9: copy into pinned memory before the transfer")
    ap.add_argument("--torch-profile", default="")
    ap.add_argument("--q", choices=["none", "body", "head", "all", "fp4h8", "fp4", "fp4_h8", "fp4_h4"], default="none",
                    help="which CoT-decode weights are quantized: FP8 body 28 layers / output layer / both; "
                         "fp4 = FP4 body (a2e_fp4q, method from A2E_FP4_METHOD, e.g. gptq:0.1) with the output layer in "
                         "BF16 (fp4), FP8 (fp4_h8) or FP4 (fp4_h4); fp4h8 = FP4 body with plain rounding (a2e_fp4) + FP8 output layer")
    ap.add_argument("--q-rows", default="", help="e.g. qkv=1,head=1 (overrides a2e_fp8.DEFAULT_ROWS)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    feat = set(f for f in args.feat.split(",") if f)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    logf = open(out.with_suffix(".log"), "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    def emit(rec):
        with open(out, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    import a2e_stagea as S

    if "p" in feat:
        S.install_processor_once()
    from alpamayo2_edge.checkpoint import load_model
    from alpamayo2_edge.helper import prepare_model_inputs, to_device

    meta = {"kind": "meta", "label": args.label, "args": vars(args), "feat": sorted(feat),
            "versions": versions(args), "temp_start": gpu_temp()}
    log(json.dumps(meta, ensure_ascii=False))

    # input
    t = time.perf_counter()
    if args.mode == "single":
        samples = [torch.load(args.input[3:], weights_only=False)]
    else:
        t0s = list(range(args.t0_start, args.t0_stop + 1, args.stride))
        if args.av2_log:  # AV2: all re-rendering is done here (outside the measured interval)
            from av2_stream import AV2StreamSource

            # --av2-log accepts multiple comma-separated logs. The input source is rebuilt for each log
            # (it holds only one log's re-rendering results. From the 2nd log on, it is built before the measured passes. All outside the measured interval)
            av2_logs = [x for x in args.av2_log.split(",") if x]

            def make_src(path):
                if args.av2_render == "inline":
                    from av2_render_fast import AV2InlineSource
                    s_ = AV2InlineSource(path, t0s, config=args.av2_config, nav=args.av2_nav, decode=args.av2_decode,
                                         out_mode=args.av2_out)
                else:
                    s_ = AV2StreamSource(path, t0s, config=args.av2_config, nav=args.av2_nav)
                emit({"kind": "av2_source", **s_.describe()})
                log(json.dumps({"kind": "av2_source", **s_.describe()}, ensure_ascii=False))
                return s_

            src = make_src(av2_logs[0])
        else:
            raise SystemExit("the public version supports continuous input only via --av2-log")
        samples = t0s
    log(f"input ready: {len(samples)} samples, {time.perf_counter() - t:.1f}s")

    t = time.perf_counter()
    model, _ = load_model(args.model, device="cuda", dtype="bfloat16", attn_implementation="sdpa",
                          local_files_only=True)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t
    log(f"model load {load_s:.1f}s")

    accel = None
    if args.accel != "off":
        from alpamayo2_edge.accel import AccelConfig, enable_acceleration

        accel = enable_acceleration(model, AccelConfig(mode=args.accel))
    qinfo = None
    if args.q != "none":
        import a2e_fp8 as Q

        if accel is None:
            raise SystemExit("--q is used on top of the official acceleration (--accel compile/tuned)")
        rows = {k: int(v) for k, v in (kv.split("=") for kv in args.q_rows.split(",") if kv)}
        if args.q in ("fp4", "fp4_h8", "fp4_h4"):  # body FP4 (a2e_fp4q), output layer BF16 / FP8 / FP4
            import os
            import a2e_fp4q as Q4Q

            qinfo = {"q": args.q, **Q4Q.install(accel, method=os.environ.get("A2E_FP4_METHOD", "rtn"),
                                                head={"fp4": "bf16", "fp4_h8": "fp8", "fp4_h4": "fp4"}[args.q])}
        elif args.q == "fp4h8":  # body 28 layers in FP4 (a2e_fp4), output layer in FP8
            import a2e_fp4 as Q4

            dec = Q.decode_accelerator(accel)
            if dec.step_graphs or dec.sampler.graphs:
                raise RuntimeError("cannot be swapped after the decode graphs have been built")
            qstep = Q4.build_step4(dec.vlm, head="fp8")
            dec.compiled_step = Q.compiled(qstep, accel)
            qinfo = {"q": args.q, "so4": Q4.SO4,
                     "body_fp4": sum(m.nbytes() for m in qstep.qlayers.modules() if isinstance(m, Q4.Q4Linear)),
                     "head_fp8": qstep.qhead.nbytes()}
        else:
            qstep = Q.install(accel, body=args.q in ("body", "all"), head=args.q in ("head", "all"), rows=rows)
            qinfo = {"q": args.q, **Q.decode_accelerator(accel)._a2e_fp8, **qstep.weight_bytes()}
        log(f"quantized decode: {qinfo}")
    vlm = model.vlm
    vacc = S.VisionAccel(vlm.model, mode=args.vit_mode) if "v" in feat else None
    cache = S.VitFeatureCache(vlm.model) if "c" in feat else None
    if cache is not None and "g" not in feat:
        raise SystemExit("c (ViT cache) must be used together with g (GPU preprocessing)")
    gprep = S.GpuPrep(model, cache=cache, pin=bool(args.pin)) if "g" in feat else None

    marks = Marks(sync=bool(args.sync_stages))
    wrap(vlm.model, "get_image_features", marks, "vision_start", "vision_end", "stage::vision")
    wrap(model, "_generate_with_shared_prefill", marks, "gen_start", "gen_end", "stage::prefill+decode")
    wrap(vlm, "generate", marks, "decode_start", "decode_end", "stage::decode")
    wrap(model, "_sample_actions", marks, "expert_start", "expert_end", "stage::expert")
    wrap(model.expert.action_space, "action_to_traj", marks, "traj_start", "traj_end", "stage::traj")
    settings = dict(top_p=0.98, top_k=None, temperature=0.6)
    tok = model.tokenizer

    def one_call(data, seed):
        marks.reset()
        marks.enabled = True
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        marks.mark("call_start")
        with torch.profiler.record_function("stage::preprocess"):
            if gprep is not None:
                prepared = gprep(data)
            else:
                prepared = prepare_model_inputs(data, model.config, model.tokenizer)
                prepared = to_device(prepared, device="cuda")
        marks.mark("pre_end")
        torch.manual_seed(seed)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            xyz, rot, _, extra = model.sample_trajectories_from_data(
                prepared, num_traj_sets=1, num_traj_samples=1, max_generation_length=args.max_gen,
                diffusion_kwargs={"inference_step": args.steps, "generator": gen}, return_extra=True,
                **settings)
        marks.mark("call_end")
        torch.cuda.synchronize()
        e2e = (time.perf_counter() - t0) * 1000
        marks.enabled = False

        def span(a, b):
            ma, mb = marks.first(a), marks.last(b)
            return None if ma is None or mb is None else ma[0].elapsed_time(mb[0])

        st = {
            "preprocess": span("call_start", "pre_end"),
            "pre_gen": span("pre_end", "gen_start"),
            "vision": span("vision_start", "vision_end"),
            "prefill_total": span("gen_start", "decode_start"),
            "decode": span("decode_start", "decode_end"),
            "gen_tail": span("decode_end", "gen_end"),
            "post_decode": span("gen_end", "expert_start"),
            "expert": span("expert_start", "expert_end"),
            "traj": span("traj_start", "traj_end"),
            "tail": span("traj_end", "call_end"),
            "gpu_total": span("call_start", "call_end"),
        }
        if st["prefill_total"] is not None and st["vision"] is not None:
            st["prefill_llm"] = st["prefill_total"] - st["vision"]
        ids = extra["generated_ids"]
        n_tok = int((ids >= 0).sum(dim=1)[0])
        gid = [int(x) for x in ids[0].tolist() if x >= 0]
        cot = extra.get("cot")
        cot = cot.reshape(-1).tolist()[0] if cot is not None else None
        return {"e2e_ms": e2e, "stages_ms": st, "cot_tokens": n_tok, "cot": cot, "gen_ids": gid,
                "nominal": bool(cot is not None and "<NOMINAL>" in cot),
                "xyz": [[round(v, 5) for v in p] for p in xyz[0, 0, 0].float().tolist()],
                "images_vit": gprep.last.get("images_processed") if gprep is not None else 22}

    def base_rec(**kw):
        r = {"kind": "run", "label": args.label}
        r.update(kw)
        return r

    if args.mode == "single":
        data = samples[0]
        temp = gpu_temp()
        if cache is not None:
            cache.clear()
        r = one_call(data, args.seed)
        emit(base_rec(idx=0, phase="first", temp_before=temp, load_s=load_s,
                      max_mem_gb=torch.cuda.max_memory_allocated() / 2**30, t0_us=data.get("t0_us"),
                      seed=args.seed, **r))
        log(f"first e2e={r['e2e_ms']:.1f} tok={r['cot_tokens']} cot={r['cot']!r}")
        for i in range(1, args.runs + 1):
            temp = thermal_gate(log)
            seed = args.seed if args.seed_mode == "fixed" else args.seed + i
            if cache is not None:
                cache.clear()
            prof = None
            ctx = contextlib.nullcontext()
            if args.torch_profile and i == args.runs:
                prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                          torch.profiler.ProfilerActivity.CUDA])
                ctx = prof
            with ctx:
                r = one_call(data, seed)
            emit(base_rec(idx=i, phase="warm", temp_before=temp, seed=seed, profiled=prof is not None,
                          t0_us=data.get("t0_us"), **r))
            log(f"run {i} e2e={r['e2e_ms']:.1f} tok={r['cot_tokens']} pre={r['stages_ms']['preprocess']:.1f} "
                f"vit={r['stages_ms']['vision']:.1f}")
            if prof is not None:
                pdir = Path(args.torch_profile)
                pdir.mkdir(parents=True, exist_ok=True)
                prof.export_chrome_trace(str(pdir / f"{args.label}_trace.json"))
                (pdir / f"{args.label}_kernels.txt").write_text(
                    prof.key_averages().table(sort_by="cuda_time_total", row_limit=60))
    else:
        # warm-up pass (compilation and graph capture, per shape) + measured passes. With multiple logs, this is repeated per log
        # (the warm-up pass from the 2nd log on is --prep-steps-next steps. Compilation is done with the 1st log, so it only fills the ViT cache)
        logs_iter = av2_logs if args.av2_log else [None]
        for li, lpath in enumerate(logs_iter):
          if li > 0:
            del src
            torch.cuda.empty_cache()
            src = make_src(lpath)
          log_id = Path(lpath).name if lpath else None
          prep_n = args.prep_steps if li == 0 else args.prep_steps_next
          for p in range(args.passes + 1):
            seed = args.seed + (0 if args.seed_mode == "fixed" else p - 1)
            if cache is not None:
                cache.clear()
            inline = args.av2_log and args.av2_render == "inline"
            if inline:
                src.reset()
            for k, t0 in enumerate(samples):
                if p == 0 and k >= prep_n:
                    break
                if inline:  # also measure re-rendering: after the temperature check, run re-rendering → inference back to back
                    temp = thermal_gate(log)
                    st0 = dict(src.r.stats)
                    torch.cuda.synchronize()
                    tr = time.perf_counter()
                    data = src.sample(t0)
                    torch.cuda.synchronize()
                    render_ms = (time.perf_counter() - tr) * 1000
                    # breakdown (until JPEGs are read and loaded onto the GPU / until the remap is issued. The remap is asynchronous, so this is approximate)
                    rdet = {k: round(src.r.stats[k] - st0[k], 2) for k in st0}
                    r = one_call(data, seed)
                    r["render_ms"] = round(render_ms, 2)
                    r["render_detail"] = rdet
                    r["total_ms"] = round(render_ms + r["e2e_ms"], 2)
                else:
                    data = src.sample(t0)  # not included in the time (corresponds to input from the sensors)
                    temp = thermal_gate(log)
                    r = one_call(data, seed)
                if p == 0:
                    phase = "prep_pass"
                elif k == 0:
                    phase = "cold"
                elif t0 < args.measure_from:
                    phase = "warmin"
                else:
                    phase = "stream"
                emit(base_rec(idx=k, pass_=p, phase=phase, seed=seed, t0_us=t0, temp_before=temp, log_id=log_id, **r))
                log(f"{(log_id or '')[:8]} pass {p} step {k} t0={data['t0_us']/1e6:.1f} e2e={r['e2e_ms']:.1f} tok={r['cot_tokens']} "
                    f"vit_imgs={r['images_vit']} pre={r['stages_ms']['preprocess']:.1f} vit={r['stages_ms']['vision']:.1f}")

    summary = {"kind": "summary", "label": args.label, "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
               "temp_end": gpu_temp()}
    if accel is not None:
        summary["accel"] = accel.describe()
    if qinfo is not None:
        summary["fp8"] = qinfo
    if vacc is not None:
        summary["vision_accel"] = vacc.describe()
    if cache is not None:
        summary["vit_cache"] = dict(cache.stats)
    emit(json.loads(json.dumps(summary, default=str)))
    log("done")


if __name__ == "__main__":
    main()

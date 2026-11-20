#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# a2e_bench2.py — の測定。a2e_bench.pyと同じ段階の区切り・同じ時計で、
# 4 つの手(a2e_stagea.py)を --feat で積み上げて測る。
#
#   --feat   空 = 公式どおり / p = #1 プロセッサを1回だけ読む(PR #11)/ g = #9 GPU で縮小
#            v = #3 ViT を compile + CUDA グラフ / c = #2 ViT の結果を画像ごとに使い回す
#   --mode single : 1 回だけ動かす測り方。毎回 ViT のキャッシュを空にして、同じサンプルを入れる
#   --mode stream : 連続して動かす測り方。同じクリップの t0 を --stride ずつ進めたサンプルを順に入れる。
#                   パスの始めにキャッシュを空にする(パスの最初の 1 歩は「冷えた」状態として別に数える)。
#
# 各呼び出しの記録: e2e(前処理込み、前後で同期した壁時計)、段階別(境界で同期した CUDA イベント)、
# CoT のトークン列と文、軌跡(候補 0 の xyz 64 点)、ViT に通した画像の枚数。
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
    # AV2: 連続入力の入力源を AV2 のログにする(av2_stream.AV2StreamSource)。t0 はログ先頭からの µs
    ap.add_argument("--av2-log", default="")
    ap.add_argument("--av2-config", default="wide_ft")
    ap.add_argument("--av2-nav", default="gt")
    # inline = 描き直し(JPEG の読み込みから)を各歩の中で行い、推論と合わせた時間も記録する
    ap.add_argument("--av2-render", choices=["pre", "inline"], default="pre")
    ap.add_argument("--av2-decode", default="pil_threads", help="inline の JPEG の読み方(av2_render_fast)")
    ap.add_argument("--av2-out", default="pinned", help="inline の描いた絵の持ち方(av2_render_fast の out_mode: cpu / pinned)")
    ap.add_argument("--t0-start", type=int, default=2_000_000, help="stream: 始める t0(公式ローダの下限 2.0 s)")
    ap.add_argument("--measure-from", type=int, default=4_000_000,
                    help="stream: この t0 から計測に数える(それより前はキャッシュを満たす期間)")
    ap.add_argument("--t0-stop", type=int, default=19_900_000)
    ap.add_argument("--prep-steps", type=int, default=30, help="stream: 準備のパスで回す歩数")
    ap.add_argument("--prep-steps-next", type=int, default=5, help="stream: 2 本目からのログの準備のパスの歩数")
    ap.add_argument("--stride", type=int, default=100_000)
    ap.add_argument("--passes", type=int, default=2, help="stream: 計測するパス数(このほかに準備のパスを 1 回)")
    ap.add_argument("--nav-text", default="Turn right in 30m")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--seed-mode", choices=["fixed", "vary"], default="fixed")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--sync-stages", type=int, default=1)
    ap.add_argument("--vit-mode", default="default", help="#3 の inductor モード")
    ap.add_argument("--pin", type=int, default=0, help="#9: 転送前に pinned メモリへ写す")
    ap.add_argument("--torch-profile", default="")
    ap.add_argument("--q", choices=["none", "body", "head", "all", "fp4h8"], default="none",
                    help="CoT デコードの重みを FP8 で読む範囲(本体 28 層 / 出力層 / 両方)")
    ap.add_argument("--q-rows", default="", help="例 qkv=1,head=1(a2e_fp8.DEFAULT_ROWS を上書き)")
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

    # 入力
    t = time.perf_counter()
    if args.mode == "single":
        samples = [torch.load(args.input[3:], weights_only=False)]
    else:
        t0s = list(range(args.t0_start, args.t0_stop + 1, args.stride))
        if args.av2_log:  # AV2: 描き直しはここ(測る区間の外)で全部済ませる
            from av2_stream import AV2StreamSource

            # --av2-log はカンマ区切りで複数のログを取れる。ログごとに入力源を作り直す
            # (1 本ぶんの描き直しの結果だけを持つ。2 本目からは計測のパスの前に作る。どれも測る区間の外)
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
            raise SystemExit("公開版の連続入力は --av2-log だけ")
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
            raise SystemExit("--q は公式の加速(--accel compile/tuned)の上で使う")
        rows = {k: int(v) for k, v in (kv.split("=") for kv in args.q_rows.split(",") if kv)}
        if args.q == "fp4h8":  # 本体 28 層を FP4(a2e_fp4)、出力層を FP8
            import a2e_fp4 as Q4

            dec = Q.decode_accelerator(accel)
            if dec.step_graphs or dec.sampler.graphs:
                raise RuntimeError("デコードのグラフを作った後には入れ替えられない")
            qstep = Q4.build_step4(dec.vlm, head="fp8")
            dec.compiled_step = Q.compiled(qstep, accel)
            qinfo = {"q": args.q, "so4": Q4.SO4,
                     "body_fp4": sum(m.nbytes() for m in qstep.qlayers.modules() if isinstance(m, Q4.Q4Linear)),
                     "head_fp8": qstep.qhead.nbytes()}
        else:
            qstep = Q.install(accel, body=args.q in ("body", "all"), head=args.q in ("head", "all"), rows=rows)
            qinfo = {"q": args.q, **Q.decode_accelerator(accel)._a2e_fp8, **qstep.weight_bytes()}
        log(f"FP8 デコード: {qinfo}")
    vlm = model.vlm
    vacc = S.VisionAccel(vlm.model, mode=args.vit_mode) if "v" in feat else None
    cache = S.VitFeatureCache(vlm.model) if "c" in feat else None
    if cache is not None and "g" not in feat:
        raise SystemExit("c (ViT cache) は g (GPU 前処理) と組で使う")
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
        # 準備のパス(コンパイルとグラフの取得、形ごと)+ 計測のパス。複数のログなら、ログごとにこれを繰り返す
        # (2 本目からの準備のパスは --prep-steps-next 歩。コンパイルは 1 本目で済んでいるので、ViT のキャッシュを満たすだけ)
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
                if inline:  # 描き直しも測る: 温度の確認のあと、描き直し → 推論を続けて動かす
                    temp = thermal_gate(log)
                    st0 = dict(src.r.stats)
                    torch.cuda.synchronize()
                    tr = time.perf_counter()
                    data = src.sample(t0)
                    torch.cuda.synchronize()
                    render_ms = (time.perf_counter() - tr) * 1000
                    # 内訳(JPEG を読んで GPU に載せるまで / 写し替えを出すまで。写し替えは非同期なので目安)
                    rdet = {k: round(src.r.stats[k] - st0[k], 2) for k in st0}
                    r = one_call(data, seed)
                    r["render_ms"] = round(render_ms, 2)
                    r["render_detail"] = rdet
                    r["total_ms"] = round(render_ms + r["e2e_ms"], 2)
                else:
                    data = src.sample(t0)  # 時間に含めない(センサからの入力に当たる)
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

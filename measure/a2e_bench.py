#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# a2e_bench.py — measures the per-stage time of Alpamayo 2 Edge
#
# Uses the inference path of the official alpamayo2_edge package (NVlabs/alpamayo2-edge) as is,
# only layering timing wrappers on the methods of the model instance; the computation itself is unchanged.
# Acceleration (--accel compile|tuned) uses the official enable_acceleration as is, with the wrappers layered outside it.
#
# Stages (aligned with the boundaries in the code):
#   preprocess   : prepare_model_inputs (CPU: prompt assembly, bicubic image resize, tokenization, normalization) + GPU transfer
#   pre_gen      : start of sample_trajectories_from_data up to the start of generation (history tokenization fuse_traj_tokens, etc.)
#   vision       : vlm.model.get_image_features (ViT 27 layers + merger/projector)
#   prefill_llm  : whole prefill − vision (embedding, rope index, 28 LLM layers over the whole prompt)
#   decode       : vlm.generate (sequential generation of CoT tokens, including sampling)
#   post_decode  : end of generation up to the start of the expert (anchor search, KV truncation, position/mask creation)
#   expert       : _sample_actions (Euler loop of flow matching, 10 steps by default)
#   traj         : action_space.action_to_traj (acceleration, curvature → trajectory integration)
#   tail         : the rest (text decoding, etc.)
#
# Clock: a CUDA event is recorded at each boundary. With --sync-stages 1 (default), synchronize at each boundary before
# recording (same as run.py in the Thor recipe: "synchronize at stage boundaries"). e2e is wall-clock time with synchronize before and after.

import argparse
import contextlib
import functools
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


def gpu_temp() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return None


def thermal_gate(log, hot=84, cool=60, rest_s=180):
    t = gpu_temp()
    if t is not None and t >= hot:
        log(f"[thermal] GPU {t}C >= {hot}C: pause {rest_s}s")
        time.sleep(rest_s)
        while True:
            t = gpu_temp()
            if t is None or t <= cool:
                break
            time.sleep(10)
        log(f"[thermal] GPU {t}C: resume")
    return t


class Marks:
    """Records named boundaries (CUDA event + perf_counter) in order."""

    def __init__(self, sync: bool):
        self.sync = sync
        self.items: list[tuple[str, torch.cuda.Event, float]] = []
        self.enabled = False

    def mark(self, name: str):
        if not self.enabled:
            return
        if self.sync:
            torch.cuda.synchronize()
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self.items.append((name, ev, time.perf_counter()))

    def reset(self):
        self.items = []

    def first(self, name):
        for n, ev, t in self.items:
            if n == name:
                return ev, t
        return None

    def last(self, name):
        for n, ev, t in reversed(self.items):
            if n == name:
                return ev, t
        return None


def wrap(obj, attr, marks: Marks, start: str, end: str, rf_name: str | None = None):
    original = getattr(obj, attr)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        marks.mark(start)
        ctx = torch.profiler.record_function(rf_name) if rf_name else contextlib.nullcontext()
        with ctx:
            out = original(*args, **kwargs)
        marks.mark(end)
        return out

    setattr(obj, attr, wrapper)
    return original


class ForceAnchorFirst:
    """Pseudo-condition "omit reasoning": forces the first generated token to the anchor <|traj_future_start|>.
    The official code has no NOMINAL branch, so this produces the state of going to the expert with 0 CoT tokens."""

    def __init__(self, anchor: int, prompt_len: int):
        self.anchor, self.prompt_len = anchor, prompt_len

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] == self.prompt_len:
            keep = scores[:, self.anchor].clone()
            scores.fill_(-math.inf)
            scores[:, self.anchor] = keep.clamp(min=0) + 1.0
        return scores


class BanFirst:
    """Pseudo-condition "produce reasoning": bans the first token of <NOMINAL> and the anchor at the first generated token.
    On synthetic input the model outputs <NOMINAL> every time, so this is used to measure the time spent generating reasoning."""

    def __init__(self, ids: list[int], prompt_len: int):
        self.ids, self.prompt_len = ids, prompt_len

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] == self.prompt_len:
            scores[:, self.ids] = -math.inf
        return scores


def synthetic_sample(nav_text: str, h: int = 1080, w: int = 1920, seed: int = 0) -> dict:
    """Synthetic input in the same form as the official loader (load_physical_aiavdataset). Images are a smooth pattern + weak noise."""
    from alpamayo2_edge.sample import CAMERA_FRAME_COUNTS, CAMERA_PIXEL_BOUNDS, validate_sample

    g = torch.Generator().manual_seed(seed)
    front = [-2_000_000, -1_000_000, -500_000, -100_000, 0]
    other = [-1_000_000, -100_000, 0]
    t0 = 4_000_000
    frames, cams, ts = [], [], []
    yy = torch.linspace(0, 1, h)[:, None]
    xx = torch.linspace(0, 1, w)[None, :]
    for cam in CAMERA_FRAME_COUNTS:
        offs = front if cam in (1, 6) else other
        for k, o in enumerate(offs):
            base = torch.stack([
                128 + 100 * torch.sin(6.28 * (xx * (cam + 1) + yy * 0.5 + k * 0.05)),
                128 + 80 * torch.cos(6.28 * (yy * 2 + xx * 0.3 * (cam + 1))),
                (yy * 200 + 30).expand(h, w),
            ])
            img = (base + 12 * torch.randn(3, h, w, generator=g)).clamp(0, 255).to(torch.uint8)
            frames.append(img[None])
            cams.append(cam)
            ts.append(t0 + o)
    tt = torch.arange(-19, 1, dtype=torch.float32) * 0.1
    xyz = torch.zeros(1, 20, 3)
    xyz[0, :, 0] = 10.0 * tt  # 10 m/s straight ahead
    rot = torch.eye(3).expand(1, 20, 3, 3).contiguous()
    data = {
        "image_frames": torch.stack(frames).contiguous(),
        "camera_indices": torch.tensor(cams, dtype=torch.int64),
        "absolute_timestamps": torch.tensor(ts, dtype=torch.int64),
        "ego_history_xyz": xyz,
        "ego_history_rot": rot,
        "nav_text": [nav_text],
        "camera_pixel_bounds": dict(CAMERA_PIXEL_BOUNDS),
        "min_pixels": 163840,
        "max_pixels": 196608,
        "input_profile": "camera_subsampled",
        "history_num_steps": 20,
        "provenance": {"dataset": "synthetic", "seed": seed, "h": h, "w": w},
    }
    validate_sample(data)
    return data


def load_input(args, log):
    if args.input == "synthetic":
        return synthetic_sample(args.nav_text)
    if args.input.startswith("pt:"):
        return torch.load(args.input[3:], weights_only=False)
    raise ValueError(args.input)


def versions(args) -> dict:
    from importlib.metadata import PackageNotFoundError, version

    v = {}
    for name in ("torch", "torchvision", "transformers", "triton", "flash-attn", "accelerate",
                 "numpy", "physical-ai-av"):
        try:
            v[name] = version(name)
        except PackageNotFoundError:
            v[name] = None
    v["torch_cuda"] = torch.version.cuda
    v["cudnn"] = torch.backends.cudnn.version()
    v["device"] = torch.cuda.get_device_name(0)
    v["capability"] = list(torch.cuda.get_device_capability(0))
    v["arch_list"] = torch.cuda.get_arch_list()
    try:
        v["driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    try:
        repo = Path(__import__("alpamayo2_edge").__file__).resolve().parents[2]
        v["alpamayo2_edge_commit"] = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        pass
    v["env"] = {k: os.environ.get(k) for k in (
        "TRITON_PTXAS_PATH", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_HOME")}
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", default="synthetic", help="synthetic | clip:<uuid>:<t0_us> | pt:<path>")
    ap.add_argument("--save-input")
    ap.add_argument("--nav-text", default="Turn right in 30m")
    ap.add_argument("--accel", choices=["off", "compile", "tuned"], default="off")
    ap.add_argument("--attn", choices=["sdpa", "flash_attention_2"], default="sdpa")
    ap.add_argument("--cot", choices=["free", "none", "reason"], default="free",
                    help="free=left to the model / none=force the anchor (CoT 0) / reason=ban the first token of <NOMINAL>")
    ap.add_argument("--num-traj-sets", type=int, default=1)
    ap.add_argument("--num-traj-samples", type=int, default=1)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--seed-mode", choices=["fixed", "vary"], default="fixed")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--sync-stages", type=int, default=1)
    ap.add_argument("--rest-every", type=int, default=0, help="check the temperature gate every N runs (0=every run)")
    ap.add_argument("--cache-processor", type=int, default=0,
                    help="1=load the processor once (equivalent to upstream PR #11). Default 0 loads it every time, as in the official code")
    ap.add_argument("--torch-profile", help="capture the last run with torch.profiler and save it to this directory")
    ap.add_argument("--out", required=True, help="jsonl")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

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

    from alpamayo2_edge.checkpoint import load_model
    from alpamayo2_edge import helper as _helper
    from alpamayo2_edge.helper import prepare_model_inputs, to_device

    if args.cache_processor:
        _orig_get_processor = _helper.get_processor
        _proc_cache = {}

        def _cached_get_processor(tokenizer, model_config):
            key = (id(tokenizer), str(getattr(model_config, "_name_or_path", "")))
            if key not in _proc_cache:
                _proc_cache[key] = _orig_get_processor(tokenizer, model_config)
            return _proc_cache[key]

        _helper.get_processor = _cached_get_processor

    meta = {"kind": "meta", "label": args.label, "args": vars(args), "versions": versions(args),
            "temp_start": gpu_temp()}
    log(json.dumps(meta, ensure_ascii=False))

    data = load_input(args, log)

    t = time.perf_counter()
    model, loading_info = load_model(args.model, device="cuda", dtype="bfloat16",
                                     attn_implementation=args.attn, local_files_only=True)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t
    log(f"model load {load_s:.1f}s")

    accel = None
    if args.accel != "off":
        from alpamayo2_edge.accel import AccelConfig, enable_acceleration

        accel = enable_acceleration(model, AccelConfig(mode=args.accel))
        log(f"accel enabled: {json.dumps(accel.describe(), default=str)[:400]}")

    marks = Marks(sync=bool(args.sync_stages))
    vlm = model.vlm
    wrap(vlm.model, "get_image_features", marks, "vision_start", "vision_end", "stage::vision")
    wrap(model, "_generate_with_shared_prefill", marks, "gen_start", "gen_end", "stage::prefill+decode")
    wrap(vlm, "generate", marks, "decode_start", "decode_end", "stage::decode")
    wrap(model, "_sample_actions", marks, "expert_start", "expert_end", "stage::expert")
    wrap(model.expert.action_space, "action_to_traj", marks, "traj_start", "traj_end", "stage::traj")

    anchor = model.config.traj_ids["future_start"]
    if args.cot in ("none", "reason"):
        orig_generate = vlm.generate
        tok = model.tokenizer
        ban = sorted({tok.convert_tokens_to_ids(t) for t in ("<N", "\u0120<", "<")} - {None, tok.unk_token_id} | {anchor})
        log(f"cot mode {args.cot}: anchor={anchor} ban_first={ban}")

        def generate_force(*a, **kw):
            from transformers.generation.logits_process import LogitsProcessorList

            plen = kw["input_ids"].shape[1]
            procs = LogitsProcessorList(list(kw.get("logits_processor") or []))
            procs.append(ForceAnchorFirst(anchor, plen) if args.cot == "none" else BanFirst(ban, plen))
            kw["logits_processor"] = procs
            return orig_generate(*a, **kw)

        vlm.generate = generate_force

    settings = dict(top_p=0.98, top_k=None, temperature=0.6)

    def one_call(seed: int, profile_dir: str | None = None):
        marks.reset()
        marks.enabled = True
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        marks.mark("call_start")
        with torch.profiler.record_function("stage::preprocess"):
            prepared = prepare_model_inputs(data, model.config, model.tokenizer)
            prepared = to_device(prepared, device="cuda")
        marks.mark("pre_end")
        torch.manual_seed(seed)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            xyz, rot, _, extra = model.sample_trajectories_from_data(
                prepared,
                num_traj_sets=args.num_traj_sets,
                num_traj_samples=args.num_traj_samples,
                max_generation_length=args.max_gen,
                diffusion_kwargs={"inference_step": args.steps, "generator": gen},
                return_extra=True,
                **settings,
            )
        marks.mark("call_end")
        torch.cuda.synchronize()
        e2e = (time.perf_counter() - t0) * 1000
        marks.enabled = False

        def span(a, b, first_a=True, last_b=True):
            ma = marks.first(a) if first_a else marks.last(a)
            mb = marks.last(b) if last_b else marks.first(b)
            if ma is None or mb is None:
                return None
            return ma[0].elapsed_time(mb[0])

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
        n_tok = (ids >= 0).sum(dim=1).tolist()
        cot = extra.get("cot")
        cot = cot.reshape(-1).tolist() if cot is not None else None
        return e2e, st, n_tok, cot, xyz

    # 1st run (first run): includes compilation, caches and CUDA graph capture
    temp = gpu_temp()
    e2e, st, n_tok, cot, xyz = one_call(args.seed)
    rec = {"kind": "run", "label": args.label, "idx": 0, "phase": "first", "e2e_ms": e2e,
           "stages_ms": st, "cot_tokens": n_tok, "cot": cot, "temp_before": temp,
           "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30, "load_s": load_s,
           "xyz_last": xyz[0, 0, 0, -1].float().tolist()}
    emit(rec)
    log(f"first e2e={e2e:.1f}ms tokens={n_tok} stages={json.dumps({k: round(v, 1) for k, v in st.items() if v is not None})}")
    log(f"cot: {cot}")

    for i in range(1, args.runs + 1):
        if args.rest_every == 0 or i % args.rest_every == 0:
            temp = thermal_gate(log)
        seed = args.seed if args.seed_mode == "fixed" else args.seed + i
        prof_ctx = contextlib.nullcontext()
        prof = None
        if args.torch_profile and i == args.runs:
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,
            )
            prof_ctx = prof
        with prof_ctx:
            e2e, st, n_tok, cot, xyz = one_call(seed)
        rec = {"kind": "run", "label": args.label, "idx": i, "phase": "warm", "seed": seed,
               "e2e_ms": e2e, "stages_ms": st, "cot_tokens": n_tok, "temp_before": temp,
               "profiled": prof is not None}
        if args.seed_mode == "vary" or i == 1:
            rec["cot"] = cot
        emit(rec)
        log(f"run {i} e2e={e2e:.1f}ms tokens={n_tok} decode={st['decode']:.1f} expert={st['expert']:.1f}")
        if prof is not None:
            pdir = Path(args.torch_profile)
            pdir.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(pdir / f"{args.label}_trace.json"))
            (pdir / f"{args.label}_kernels.txt").write_text(
                prof.key_averages().table(sort_by="cuda_time_total", row_limit=60))
            log(f"profile saved to {pdir}")

    summary = {"kind": "summary", "label": args.label,
               "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
               "temp_end": gpu_temp()}
    if accel is not None:
        summary["accel"] = accel.describe()
    emit(json.loads(json.dumps(summary, default=str)))
    log("done")


if __name__ == "__main__":
    main()

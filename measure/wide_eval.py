#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# wide_eval.py — 広いデータで、デコードの実装(重みの形式)ごとの出力を同じ場面・同じ seed でペアにして記録する
#
# 1 つのプロセスで、デコードの実装(変種)を切り替えながら同じ入力を流す。前処理・ViT・プリフィル・expert は共通(BF16)。
# 変種は VARIANTS に名前と作り方を足すだけで増やせる(--variant-module で外のファイルから足すこともできる)。
#   base    : 公式 tuned(BF16)+ 加速の 4 つの手。比べる基準
#   eager   : 同じ BF16 の DecodeStep を compile せずに動かす(BF16 の丸めの揺れの幅。teacher forcing の物差し用)
#   fp8body : 本体 28 層を FP8
#   fp8all  : 本体 + 出力層を FP8( からの出発点)
#
# 1 件(clip, t0)ごとに記録するもの:
#  S. 公式のサンプリング(温度 0.6、top_p 0.98、1 候補、拡散 10 ステップ)を、--sample-variants × seeds で。
#     CoT(文とトークン)、軌跡(64 点)、時間(1 回だけの測り方: 毎回 ViT のキャッシュを空にする)、デコードの時間とトークン数。
#     拡散の乱数は seed と同じ値(公式の推論と同じ)。BF16 同士の seed の幅も、この base の seed 違いで測る。
#  T. teacher forcing: base の seed[0] の CoT を固定の入力にして、各変種の 1 トークンずつの分布を base と比べる
#     (KL(base‖変種) の温度 1 と 0.6、上位 1 位の一致)。同じ CoT・同じ拡散の乱数で expert まで回したときの軌跡の差と、
#     base で拡散の乱数だけを変えたときの軌跡の差(物差し)。
#  G. 記録された未来の自車軌跡(egomotion、t0 の自車座標、0.1〜6.4 s の 64 点)。モデルには見せない。
#
# 途中で止まっても、--out にある sample_id は飛ばして続きから回せる。
import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import a2e_fp8 as Q  # noqa: E402
import a2e_stagea as S  # noqa: E402
from qcheck import ForcedSampler, dist_metrics  # noqa: E402


# ---------------------------------------------------------------- 変種(デコードの実装)の一覧
def _v_eager(vlm, handle):
    from alpamayo2_edge.accel.decode import DecodeStep

    return DecodeStep(vlm)


def _v_fp8(body, head):
    def make(vlm, handle):
        return Q.compiled(Q.build_step(vlm, body=body, head=head), handle)

    return make


VARIANTS = {
    "eager": _v_eager,
    "fp8body": _v_fp8(True, False),
    "fp8all": _v_fp8(True, True),
    "fp8head": _v_fp8(False, True),
}


def load_variant_module(path):
    """外のファイルの VARIANTS(名前 → (vlm, handle) を受けて compiled_step を返す関数)を足す。4 ビットなどの追加用。"""
    spec = importlib.util.spec_from_file_location("extra_variants", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # torch.compile(dynamo)がモジュールを名前で探すため(無いと ModuleNotFoundError)
    spec.loader.exec_module(mod)
    VARIANTS.update(mod.VARIANTS)


# ---------------------------------------------------------------- 補助
def ade(a, b):
    d = [math.dist(p, r) for p, r in zip(a, b)]
    return sum(d) / len(d)


def gpu_temp():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=10).stdout.strip())
    except Exception:
        return None


def thermal_gate(log, hot=84, resume=60, pause=180):
    t = gpu_temp()
    if t is not None and t >= hot:
        log(f"GPU {t}℃ → {pause}s 休止、{resume}℃ まで待つ")
        time.sleep(pause)
        while (t := gpu_temp()) is not None and t > resume:
            time.sleep(10)
        log(f"再開 {t}℃")
    return t


def gt_future(avdi, clip, t0):
    """記録された未来の自車軌跡(t0 の自車座標、0.1〜6.4 s の 64 点)。範囲外なら None。"""
    ego = avdi.get_clip_feature(clip, avdi.features.LABELS.EGOMOTION)
    ts = t0 + np.arange(0, 6_400_001, 100_000, dtype=np.int64)
    lo, hi = ego.time_range
    if ts[0] < lo or ts[-1] > hi:
        return None
    st = ego(ts)
    xyz = np.asarray(st.pose.translation)
    local = st.pose.rotation[0].inv().apply(xyz - xyz[0])
    return np.round(local[1:], 4).tolist()


def graph_count(handle):
    n = 0
    for acc in handle.accelerators:
        for name in ("graphs", "step_graphs", "prefill_graphs"):
            d = getattr(acc, name, None)
            if isinstance(d, dict):
                n += len(d)
        sampler = getattr(acc, "sampler", None)
        if sampler is not None and isinstance(getattr(sampler, "graphs", None), dict):
            n += len(sampler.graphs)
    return n


class DecodeTimer:
    """vlm.generate(CoT のデコード)の前後に CUDA イベントを打つ。"""

    def __init__(self, vlm):
        self.vlm, self.orig = vlm, vlm.generate
        self.ev = None

        def gen(*a, **k):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            out = self.orig(*a, **k)
            e.record()
            self.ev = (s, e)
            return out

        vlm.generate = gen

    def ms(self):
        return None if self.ev is None else self.ev[0].elapsed_time(self.ev[1])


# ---------------------------------------------------------------- AV2: 入力と正解
def av2_scene(row, config):
    """scenes_t0.csv の 1 行 → (公式ローダと同じ形の入力, ナビ文の情報, 正解の軌跡)。"""
    import av2_build as B
    import av2_nav as N

    lg = B.AV2Log(row.data_dir)
    t0 = int(row.t0_ns)
    info = N.nav_from_future(lg, t0)
    data = B.build_sample(lg, t0, config, info["text"])
    data.pop("av2", None)
    return data, info, np.round(B.gt_future(lg, t0), 4).tolist()


# ---------------------------------------------------------------- 本体
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--scenes", required=True, help="select_scenes.py の scenes.csv")
    ap.add_argument("--data-dir", default="", help="(AV2: 使わない。場所は scenes の data_dir 列)")
    ap.add_argument("--variants", default="base,fp8body,fp8all", help="サンプリングで比べる変種(先頭は base)")
    ap.add_argument("--tf-variants", default="eager,fp8body,fp8all", help="teacher forcing で base と比べる変種")
    ap.add_argument("--variant-module", default="", help="VARIANTS を足す Python ファイル")
    ap.add_argument("--seeds", default="43,44,45")
    ap.add_argument("--noise-seed", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="先頭から n 件だけ(試運転用)")
    ap.add_argument("--so", default="")
    ap.add_argument("--out", required=True)
    # AV2: 入力の組み立て方と、参考の「いつも直進」のナビ文で BF16 だけ回す seed
    ap.add_argument("--av2-config", default="wide_ft")
    ap.add_argument("--straight-seeds", default="43,44,45,46,47", help="空なら回さない")
    args = ap.parse_args()
    if args.variant_module:
        load_variant_module(args.variant_module)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def emit(rec):
        with open(out, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    import pandas as pd

    scenes = pd.read_csv(args.scenes)
    scenes["sample_id"] = scenes["scene_id"]  # AV2: 場面の ID
    scenes["clip_id"] = scenes["log_id"]
    if args.limit:
        scenes = scenes.head(args.limit)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if r.get("kind") == "sample":
                done.add(r["sample_id"])
    seeds = [int(x) for x in args.seeds.split(",")]
    variants = args.variants.split(",")
    tf_variants = [v for v in args.tf_variants.split(",") if v]
    assert variants[0] == "base"
    if args.so:
        Q.load_ops(args.so)

    S.install_processor_once()
    from alpamayo2_edge.accel import AccelConfig, enable_acceleration
    from alpamayo2_edge.checkpoint import load_model

    model, _ = load_model(args.model, device="cuda", dtype="bfloat16", attn_implementation="sdpa", local_files_only=True)
    handle = enable_acceleration(model, AccelConfig(mode="tuned"))
    vlm = model.vlm
    S.VisionAccel(vlm.model)
    vcache = S.VitFeatureCache(vlm.model)
    gprep = S.GpuPrep(model, cache=vcache)
    V = Q.DecodeVariants(handle)
    for name in dict.fromkeys(variants[1:] + tf_variants):
        V.add(name, VARIANTS[name](vlm, handle))
    timer = DecodeTimer(vlm)
    forced = ForcedSampler(V.dec, keep=("MaskDiscreteTrajectoryLogitsProcessor", "MaskTokenIdsLogitsProcessor"))
    emit({"kind": "meta", "time": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(args), "variants": variants,
          "tf_variants": tf_variants, "so": Q._loaded.get("so"), "rows": Q.DEFAULT_ROWS,
          "n_scenes": len(scenes), "resumed_skip": len(done), "input": "AV2"})

    def call(data, seed, *, noise=None, use_forced=None, timed=False):
        if timed:
            vcache.clear()  # 1 回だけの測り方: ViT を毎回計算する
        g0 = graph_count(handle)
        torch.cuda.synchronize()
        t = time.perf_counter()
        prepared = gprep(data)
        torch.manual_seed(seed)
        gen = torch.Generator(device="cuda").manual_seed(noise if noise is not None else seed)
        saved = V.dec.sampler
        if use_forced is not None:
            forced.tokens = use_forced
            V.dec.sampler = forced
        timer.ev = None
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                xyz, _, _, extra = model.sample_trajectories_from_data(
                    prepared, num_traj_sets=1, num_traj_samples=1, max_generation_length=256,
                    diffusion_kwargs={"inference_step": 10, "generator": gen}, return_extra=True,
                    top_p=0.98, top_k=None, temperature=0.6)
        finally:
            V.dec.sampler = saved
        torch.cuda.synchronize()
        e2e = (time.perf_counter() - t) * 1000
        ids = [int(x) for x in extra["generated_ids"][0].tolist() if x >= 0]
        cot = extra.get("cot")
        cot = cot.reshape(-1).tolist()[0] if cot is not None else None
        r = {"gen_ids": ids, "cot": cot, "n_tok": len(ids),
             "xyz": [[round(v, 4) for v in p] for p in xyz[0, 0, 0].float().tolist()]}
        if timed:
            r.update(e2e_ms=round(e2e, 2), decode_ms=round(timer.ms() or 0.0, 2),
                     new_graphs=graph_count(handle) - g0)
        return r

    # ウォームアップ: 変種ごとに 1 回(コンパイルとグラフの取得。記録しない)
    first = scenes.iloc[0]
    d0, _, _ = av2_scene(first, args.av2_config)  # AV2
    tw = time.perf_counter()
    for v in dict.fromkeys(variants + tf_variants):
        V.use(v)
        call(d0, seeds[0], timed=True)
        if v in tf_variants or v == "base":
            ref = call(d0, seeds[0])["gen_ids"]
            call(d0, seeds[0], noise=args.noise_seed, use_forced=ref)
    log(f"ウォームアップ {time.perf_counter() - tw:.0f}s")
    del d0

    tstart = time.perf_counter()
    n_new = 0
    for clip, grp in scenes.groupby("clip_id", sort=False):
        todo = [r for r in grp.itertuples() if r.sample_id not in done]
        if not todo:
            continue
        straight_seeds = [int(x) for x in args.straight_seeds.split(",") if x]
        for r in todo:
            tl = time.perf_counter()
            data, info, gt = av2_scene(r, args.av2_config)  # AV2: 描き直しは時間に含めない
            load_s = time.perf_counter() - tl
            rec = {"kind": "sample", "sample_id": r.sample_id, "clip_id": clip, "split": r.split, "t0_ns": int(r.t0_ns),
                   "t0_us": int(data["t0_us"]), "motion": r.kind, "city": r.city, "nav_text": info["text"],
                   "nav_info": info, "gt_xyz": gt, "clip_load_s": round(load_s, 2), "sampled": {}, "tf": {}}
            temps = []
            # S. サンプリング(seed ごとに、変種を続けて回す: 温度の条件を変種どうしでそろえる)
            for s in seeds:
                for v in variants:
                    temps.append(thermal_gate(log))
                    V.use(v)
                    rec["sampled"].setdefault(v, {})[str(s)] = call(data, s, timed=True)
            # T. teacher forcing(base の seed[0] の CoT を固定)
            ids = rec["sampled"]["base"][str(seeds[0])]["gen_ids"]
            V.use("base")
            rb = call(data, seeds[0], noise=args.noise_seed, use_forced=ids)
            sb = forced.scores
            assert rb["gen_ids"] == ids, "forced の結果が固定した CoT と違う"
            rb2 = call(data, seeds[0], noise=args.noise_seed + 1, use_forced=ids)
            rec["tf"] = {"ref": f"base/s{seeds[0]}", "n_tok": len(ids),
                         "base_noise_alt_ade": ade(rb2["xyz"], rb["xyz"]), "vs": {}}
            for v in tf_variants:
                V.use(v)
                rv = call(data, seeds[0], noise=args.noise_seed, use_forced=ids)
                m = dist_metrics(sb, forced.scores, ids)
                m["traj_ade_vs_base"] = ade(rv["xyz"], rb["xyz"])
                rec["tf"]["vs"][v] = m
            # 参考(AV2): BF16 で、ナビ文をいつも "Continue straight" にしたとき
            if straight_seeds:
                V.use("base")
                d2 = dict(data, nav_text=["Continue straight"])
                rec["straight"] = {str(s_): call(d2, s_) for s_ in straight_seeds}
            rec["temp_max"] = max((t for t in temps if t is not None), default=None)
            emit(rec)
            n_new += 1
            b = rec["sampled"]["base"]
            line = [f"[{len(done) + n_new}/{len(scenes)}] {r.sample_id} {r.kind} {rec["nav_text"]!r} tok={b[str(seeds[0])]['n_tok']}"]
            for v in variants[1:]:
                same = sum(rec["sampled"][v][str(s)]["gen_ids"] == b[str(s)]["gen_ids"] for s in seeds)
                d = [ade(rec["sampled"][v][str(s)]["xyz"], b[str(s)]["xyz"]) for s in seeds]
                kl = rec["tf"]["vs"].get(v, {}).get("kl_t1")
                line.append(f"{v}: CoT同={same}/{len(seeds)} 軌跡差={np.median(d):.3f}m"
                            + (f" KL={np.mean(kl):.1e}" if kl else ""))
            log(" | ".join(line) + f" {rec['temp_max']}℃ ({time.perf_counter() - tstart:.0f}s)")
    emit({"kind": "summary", "seconds": time.perf_counter() - tstart, "n_new": n_new,
          "accel": json.loads(json.dumps(handle.describe(), default=str)),
          "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30})
    log("done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# summarize_speed.py <raw/speed> <out.md> <out.json> [render.json]
#   1 回だけ: phase=warm(プロファイルの回を除く)の e2e の中央値・四分位・最小〜最大、初回(phase=first)、段階ごとの中央値、
#             1 トークンあたり = decode 段階 ÷ CoT のトークン数
#   連続: phase=stream の歩。e2e の中央値・p10・p90・最大、1 秒以内(e2e < 1000 ms)の割合と 1 秒を超えた割合、
#         理由あり / NOMINAL 別、冷えた初歩(phase=cold)、段階ごとの中央値、トークン数。
#         e2e ≈ a + b × トークン を直線で当てはめ、1000 ms に収まるトークン数の上限 = (1000 − a) / b。
#         decode 段階をトークン数に当てはめた傾き = 1 トークンあたりの時間。
#   FP8 と BF16 の一致(連続 10Hz、同じ (t0, seed)): CoT が同じ割合、軌跡の差(ADE)。比べる幅 = BF16 どうしで seed だけ違う組。
import json
import math
import statistics as st
import sys
from pathlib import Path

STAGES = ["preprocess", "vision", "prefill_llm", "decode", "expert", "traj"]


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def f(x, nd=1):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def load(d):
    runs = {}
    for p in sorted(Path(d).glob("*.jsonl")):
        rs = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        runs[p.stem] = {"meta": next((r for r in rs if r.get("kind") == "meta"), {}),
                        "src": next((r for r in rs if r.get("kind") == "av2_source"), None),
                        "runs": [r for r in rs if r.get("kind") == "run"],
                        "summary": next((r for r in rs if r.get("kind") == "summary"), None)}
    return runs


def fit(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None, None, None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    res = [y - (a + b * x) for x, y in zip(xs, ys)]
    return a, b, st.pstdev(res)


def ade(a, b):
    d = [math.dist(p[:2], r[:2]) for p, r in zip(a, b)]
    return sum(d) / len(d)


def main(d, out_md, out_json, render_json=None):
    R = load(d)
    J = {"single": {}, "stream": {}, "fit": {}, "agree": {}}
    L = ["#  速さの集計(AV2、ログ 9e9bcfb7、wide_ft、BF16 / FP8)\n",
         "- 時間はすべて ms。e2e は前処理込みの壁時計(前後で同期)。段階は境界で同期した CUDA イベント。",
         "- 仮想カメラへの描き直しは入力の準備で済ませてあり、e2e に入っていない(下の「描き直し」の節)。\n",
         "## 1 回だけ(t0 = 8.5 s、初回 1 + 安定 20 回)\n",
         "| 条件 | seed | トークン | e2e 中央値 | 四分位 | 最小〜最大 | 初回 | " + " | ".join(STAGES) + " | ms/トークン |",
         "|---" * (8 + len(STAGES)) + "|"]
    for name in sorted(k for k in R if k[0] in "ABC"):
        c = R[name]
        warm = [r for r in c["runs"] if r["phase"] == "warm" and not r.get("profiled")]
        first = next((r for r in c["runs"] if r["phase"] == "first"), None)
        if not warm:
            continue
        e = [r["e2e_ms"] for r in warm]
        stg = {s: st.median([r["stages_ms"][s] for r in warm if r["stages_ms"].get(s) is not None])
               if any(r["stages_ms"].get(s) is not None for r in warm) else None for s in STAGES}
        tok = st.median([r["cot_tokens"] for r in warm])
        mpt = stg["decode"] / tok if stg["decode"] and tok else None
        J["single"][name] = {"n": len(warm), "tokens": tok, "e2e_median": st.median(e), "q25": q(e, .25),
                             "q75": q(e, .75), "min": min(e), "max": max(e),
                             "first": first["e2e_ms"] if first else None, "stages": stg, "ms_per_token": mpt,
                             "cot": warm[0].get("cot")}
        L.append(f"| {name} | {warm[0]['seed']} | {f(tok, 0)} | **{f(st.median(e))}** | {f(q(e, .25))}–{f(q(e, .75))} | "
                 f"{f(min(e))}–{f(max(e))} | {f(first['e2e_ms'] if first else None)} | "
                 + " | ".join(f(stg[s]) for s in STAGES) + f" | {f(mpt, 2)} |")
    L += ["", "## 連続(計測に数える歩 = t0 ≥ 4.0 s)\n",
          "| 条件 | 歩数 | ViT に通した枚数(中央値) | e2e 中央値 | p10 | p90 | 最大 | 1 秒を超えた割合 | 理由あり: 中央値 / 超え | NOMINAL: 中央値 / 超え(歩数) | 冷えた初歩 | トークン 中央値 / p90 / 最大 |",
          "|---" * 12 + "|"]
    for name in sorted(k for k in R if k.startswith("S")):
        c = R[name]
        rs = [r for r in c["runs"] if r["phase"] == "stream"]
        if not rs:
            continue
        e = [r["e2e_ms"] for r in rs]
        rea = [r["e2e_ms"] for r in rs if not r["nominal"]]
        nom = [r["e2e_ms"] for r in rs if r["nominal"]]
        over = lambda xs: sum(x >= 1000 for x in xs) / len(xs) if xs else float("nan")  # noqa: E731
        cold = [r["e2e_ms"] for r in c["runs"] if r["phase"] == "cold"]
        toks = [r["cot_tokens"] for r in rs]
        stg = {s: st.median([r["stages_ms"][s] for r in rs if r["stages_ms"].get(s) is not None])
               if any(r["stages_ms"].get(s) is not None for r in rs) else None for s in STAGES}
        J["stream"][name] = {"n": len(rs), "images_vit": st.median([r["images_vit"] for r in rs]),
                             "e2e_median": st.median(e), "p10": q(e, .1), "p90": q(e, .9), "max": max(e),
                             "frac_over_1s": over(e), "reason_n": len(rea),
                             "reason_median": st.median(rea) if rea else None, "reason_over_1s": over(rea),
                             "nominal_n": len(nom), "nominal_median": st.median(nom) if nom else None,
                             "nominal_over_1s": over(nom), "cold": cold, "stages": stg,
                             "tokens_median": st.median(toks), "tokens_p90": q(toks, .9), "tokens_max": max(toks),
                             "source": c["src"]}
        L.append(f"| {name} | {len(rs)} | {f(J['stream'][name]['images_vit'], 0)} | **{f(st.median(e))}** | {f(q(e, .1))} | "
                 f"{f(q(e, .9))} | {f(max(e))} | **{100 * over(e):.1f}%** | {f(st.median(rea) if rea else None)} / "
                 f"{100 * over(rea):.1f}% | {f(st.median(nom) if nom else None)} / "
                 f"{(100 * over(nom)) if nom else float('nan'):.1f}% ({len(nom)}) | {f(st.median(cold) if cold else None)} | "
                 f"{f(st.median(toks), 0)} / {f(q(toks, .9), 0)} / {max(toks)} |")
    L += ["", "### 連続の歩の段階ごとの中央値\n", "| 条件 | " + " | ".join(STAGES) + " |", "|---" * (1 + len(STAGES)) + "|"]
    for name, v in J["stream"].items():
        L.append(f"| {name} | " + " | ".join(f(v["stages"][s]) for s in STAGES) + " |")
    L += ["", "### トークン数への当てはめ(連続の歩)\n",
          "| 条件 | e2e ≈ a + b × トークン | 1000 ms に収まるトークン数の上限 | 残差の標準偏差 | decode の傾き(ms/トークン) |",
          "|---|---|---|---|---|"]
    for name in J["stream"]:
        rs = [r for r in R[name]["runs"] if r["phase"] == "stream"]
        a, b, sd = fit([r["cot_tokens"] for r in rs], [r["e2e_ms"] for r in rs])
        _, bd, _ = fit([r["cot_tokens"] for r in rs], [r["stages_ms"]["decode"] for r in rs])
        if a is None:
            continue
        J["fit"][name] = {"a": a, "b": b, "resid_sd": sd, "max_tokens_1s": (1000 - a) / b if b else None,
                          "decode_slope": bd}
        L.append(f"| {name} | {f(a)} + {f(b, 2)} × n | {f((1000 - a) / b if b else None)} | {f(sd)} | {f(bd, 2)} |")
    # FP8 と BF16 の一致(連続 10Hz、同じ (t0, seed))
    def keyed(name):
        return {(r["t0_us"], r["seed"]): r for r in R[name]["runs"] if r["phase"] == "stream"} if name in R else {}

    base = keyed("S10_pgcv")
    if base:
        L += ["", "## 連続 10Hz での出力の一致(同じ t0・同じ seed で BF16 と比べる)\n",
              "| 比べる組 | 組の数 | CoT が同じ | 軌跡の差 ADE 中央値 / p90 / 最大(m) |", "|---|---|---|---|"]
        for name in ("S10_body", "S10_all"):
            other = keyed(name)
            ks = sorted(set(base) & set(other))
            if not ks:
                continue
            sc = sum(base[k]["gen_ids"] == other[k]["gen_ids"] for k in ks) / len(ks)
            ad = [ade(base[k]["xyz"], other[k]["xyz"]) for k in ks]
            J["agree"][name] = {"n": len(ks), "same_cot": sc, "ade_median": st.median(ad), "ade_p90": q(ad, .9),
                                "ade_max": max(ad)}
            L.append(f"| BF16 対 {name} | {len(ks)} | {100 * sc:.1f}% | {f(st.median(ad), 3)} / {f(q(ad, .9), 3)} / {f(max(ad), 3)} |")
        # 比べる幅: BF16 どうしで seed だけ違う(同じ t0 の隣り合う seed)
        by_t0 = {}
        for (t0, s), r in base.items():
            by_t0.setdefault(t0, {})[s] = r
        pairs = [(v[s], v[s + 1]) for v in by_t0.values() for s in v if s + 1 in v]
        if pairs:
            sc = sum(a["gen_ids"] == b["gen_ids"] for a, b in pairs) / len(pairs)
            ad = [ade(a["xyz"], b["xyz"]) for a, b in pairs]
            J["agree"]["bf16_seed_diff"] = {"n": len(pairs), "same_cot": sc, "ade_median": st.median(ad),
                                            "ade_p90": q(ad, .9), "ade_max": max(ad)}
            L.append(f"| (幅)BF16 どうしで seed 違い | {len(pairs)} | {100 * sc:.1f}% | {f(st.median(ad), 3)} / {f(q(ad, .9), 3)} / {f(max(ad), 3)} |")
    if render_json and Path(render_json).exists():
        rj = json.loads(Path(render_json).read_text())
        J["render"] = rj
        L += ["", "## 描き直し(仮想カメラへの写し替え。推論の時間には入れていない)\n",
              f"- 写し方の表を作る(キャリブレーションごとに 1 回): {rj['maps_build_s']} s",
              f"- 22 枚を全部描く: 中央値 {rj['full_22img_ms']['median']} ms(最大 {rj['full_22img_ms']['max']})。"
              f"そのうち JPEG の読み込み {rj['jpeg_decode_ms_in_full']['median']} ms({rj['jpeg_decode_ms_in_full']['files_per_t0']} 枚)",
              f"- 連続 10Hz で増える分だけ描く: 中央値 {rj['stream10hz']['per_t0_ms_median']} ms(最大 {rj['stream10hz']['per_t0_ms_max']}、"
              f"新しく描く枚数 {rj['stream10hz']['new_images_per_t0']})",
              f"- 連続 1Hz で増える分だけ描く: 中央値 {rj['stream1hz']['per_t0_ms_median']} ms(最大 {rj['stream1hz']['per_t0_ms_max']}、"
              f"新しく描く枚数 {rj['stream1hz']['new_images_per_t0']})"]
    Path(out_md).write_text("\n".join(L) + "\n")
    Path(out_json).write_text(json.dumps(J, ensure_ascii=False, indent=1, default=str))
    print("\n".join(L))


if __name__ == "__main__":
    main(*sys.argv[1:])

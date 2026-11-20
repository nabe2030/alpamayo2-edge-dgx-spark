#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# summarize_speed.py <raw/speed> <out.md> <out.json> [render.json]
#   single input: e2e median, quartiles, min–max of phase=warm (excluding profiled runs), first run (phase=first), per-stage medians,
#             per token = decode stage ÷ number of CoT tokens
#   continuous: steps with phase=stream. e2e median, p10, p90, max, share within 1 s (e2e < 1000 ms) and share over 1 s,
#         split by with-reasoning / NOMINAL, cold first step (phase=cold), per-stage medians, token counts.
#         Linear fit e2e ≈ a + b × tokens; max tokens that fit in 1000 ms = (1000 − a) / b.
#         Slope of the decode stage fitted against token count = time per token.
#   FP8 vs BF16 agreement (continuous 10Hz, same (t0, seed)): share with identical CoT, trajectory difference (ADE). Reference spread = BF16 pairs differing only in seed.
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
    L = ["#  Speed summary (AV2, log 9e9bcfb7, wide_ft, BF16 / FP8)\n",
         "- All times in ms. e2e is wall-clock time including preprocessing (synchronized before and after). Stages are CUDA events synchronized at the boundaries.",
         "- Re-rendering to the virtual cameras is done during input preparation and is not included in e2e (see the \"Re-rendering\" section below).\n",
         "## Single input (t0 = 8.5 s, 1 first run + 20 steady-state runs)\n",
         "| Condition | seed | Tokens | e2e median | Quartiles | Min–max | First run | " + " | ".join(STAGES) + " | ms per token |",
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
    L += ["", "## Continuous (steps counted = t0 ≥ 4.0 s)\n",
          "| Condition | Steps | Images through ViT (median) | e2e median | p10 | p90 | Max | Share over 1 s | With reasoning: median / over | NOMINAL: median / over (steps) | Cold first step | Tokens median / p90 / max |",
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
    L += ["", "### Per-stage medians of continuous steps\n", "| Condition | " + " | ".join(STAGES) + " |", "|---" * (1 + len(STAGES)) + "|"]
    for name, v in J["stream"].items():
        L.append(f"| {name} | " + " | ".join(f(v["stages"][s]) for s in STAGES) + " |")
    L += ["", "### Fit against token count (continuous steps)\n",
          "| Condition | e2e ≈ a + b × tokens | Max tokens within 1000 ms | Residual SD | Decode slope (ms per token) |",
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
    # FP8 vs BF16 agreement (continuous 10Hz, same (t0, seed))
    def keyed(name):
        return {(r["t0_us"], r["seed"]): r for r in R[name]["runs"] if r["phase"] == "stream"} if name in R else {}

    base = keyed("S10_pgcv")
    if base:
        L += ["", "## Output agreement at continuous 10Hz (compared with BF16 at the same t0 and seed)\n",
              "| Pair compared | Pairs | Same CoT | Trajectory difference ADE median / p90 / max (m) |", "|---|---|---|---|"]
        for name in ("S10_body", "S10_all"):
            other = keyed(name)
            ks = sorted(set(base) & set(other))
            if not ks:
                continue
            sc = sum(base[k]["gen_ids"] == other[k]["gen_ids"] for k in ks) / len(ks)
            ad = [ade(base[k]["xyz"], other[k]["xyz"]) for k in ks]
            J["agree"][name] = {"n": len(ks), "same_cot": sc, "ade_median": st.median(ad), "ade_p90": q(ad, .9),
                                "ade_max": max(ad)}
            L.append(f"| BF16 vs {name} | {len(ks)} | {100 * sc:.1f}% | {f(st.median(ad), 3)} / {f(q(ad, .9), 3)} / {f(max(ad), 3)} |")
        # Reference spread: BF16 pairs differing only in seed (adjacent seeds at the same t0)
        by_t0 = {}
        for (t0, s), r in base.items():
            by_t0.setdefault(t0, {})[s] = r
        pairs = [(v[s], v[s + 1]) for v in by_t0.values() for s in v if s + 1 in v]
        if pairs:
            sc = sum(a["gen_ids"] == b["gen_ids"] for a, b in pairs) / len(pairs)
            ad = [ade(a["xyz"], b["xyz"]) for a, b in pairs]
            J["agree"]["bf16_seed_diff"] = {"n": len(pairs), "same_cot": sc, "ade_median": st.median(ad),
                                            "ade_p90": q(ad, .9), "ade_max": max(ad)}
            L.append(f"| (spread) BF16 vs BF16, different seed | {len(pairs)} | {100 * sc:.1f}% | {f(st.median(ad), 3)} / {f(q(ad, .9), 3)} / {f(max(ad), 3)} |")
    if render_json and Path(render_json).exists():
        rj = json.loads(Path(render_json).read_text())
        J["render"] = rj
        L += ["", "## Re-rendering (remapping to the virtual cameras; not included in inference time)\n",
              f"- Build the remap tables (once per calibration): {rj['maps_build_s']} s",
              f"- Render all 22 images: median {rj['full_22img_ms']['median']} ms (max {rj['full_22img_ms']['max']}). "
              f"Of this, JPEG decoding {rj['jpeg_decode_ms_in_full']['median']} ms ({rj['jpeg_decode_ms_in_full']['files_per_t0']} images)",
              f"- Continuous 10Hz, render only the new images: median {rj['stream10hz']['per_t0_ms_median']} ms (max {rj['stream10hz']['per_t0_ms_max']}, "
              f"new images rendered {rj['stream10hz']['new_images_per_t0']})",
              f"- Continuous 1Hz, render only the new images: median {rj['stream1hz']['per_t0_ms_median']} ms (max {rj['stream1hz']['per_t0_ms_max']}, "
              f"new images rendered {rj['stream1hz']['new_images_per_t0']})"]
    Path(out_md).write_text("\n".join(L) + "\n")
    Path(out_json).write_text(json.dumps(J, ensure_ascii=False, indent=1, default=str))
    print("\n".join(L))


if __name__ == "__main__":
    main(*sys.argv[1:])

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030 (modifications)
# SPDX-License-Identifier: Apache-2.0
#
# This file contains code modified from NVlabs/alpamayo2-edge (https://github.com/NVlabs/alpamayo2-edge,
# commit 02f55a7), licensed under the Apache License, Version 2.0 (see LICENSES/Apache-2.0.txt).
# Modified in 2026: CalibStep.forward follows the CoT decode step of alpamayo2_edge/accel/decode.py (BF16, unchanged
# numerics) and additionally accumulates the statistics H = sum x x^T of the inputs of every projection.
#
# a2e_fp4q.py — FP4 (NVFP4-like) weight-only decode for the Alpamayo 2 Edge CoT, with several ways of
# choosing the 4-bit values. The packed format and the GEMV kernel are the same as a2e_fp4.py
# (E2M1 x2 per byte, one E4M3 scale per 16 weights, one FP32 scale per row); only the rounding differs.
#
#   rtn   : round-to-nearest, block scale = amax/6 (same as a2e_fp4.quantize_nvfp4)
#   mse   : per-block scale search (grid around amax/6), minimizing the squared error weighted by
#           diag(H) when calibration statistics are available (else unweighted)
#   gptq  : GPTQ (Frantar et al.) error compensation with H = sum x x^T collected on AV2 calibration
#           scenes during BF16 CoT decoding; block scales are chosen when each 16-column group starts
#   gptqm : gptq + the mse scale search for every 16-column group
#
# Recommended configuration in the article: method "gptq:0.1" (GPTQ, damping 0.1 * mean(diag H)), output layer BF16,
# with H collected on 48 AV2 train logs that are not in the 400 evaluation scenes (measure/run_calib_fp4.sh).
#
# Calibration statistics: run wide_eval.py with the "calib" variant (BF16, records H for the inputs of
# qkv / o / fc1 / fc2 of every layer at every decode step) and A2E_CALIB_OUT=<path>; H is saved at exit.
# Quantized weights are cached in A2E_FP4_CACHE (one file per method, e.g. body_gptq:0.1.pt, about 795 MB) so later
# runs do not redo GPTQ (65 s for the 28 layers on GB10).
#
# Use from a script (before the first decode call):
#   import a2e_fp4q; a2e_fp4q.install(handle, method="gptq:0.1", head="bf16")   # handle: official enable_acceleration()
# or through measure/a2e_bench2.py --q fp4 with A2E_FP4_METHOD=gptq:0.1 and A2E_CALIB_IN=<H file>.
#
# Variants for wide_eval.py --variant-module:
#   fp4<method>[_d<pct>][_h8|_h4]  body 28 layers in FP4; GPTQ damping pct/100 (default 0.01);
#                          head BF16 (default), FP8 (_h8) or FP4 (_h4, rtn/mse rounding)
#   calib                  BF16 decode step that accumulates H (for calibration only)
import atexit
import os
from pathlib import Path

import torch

import a2e_fp8 as Q
import a2e_fp4 as F

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
SCALE_GRID = [0.62 + 0.02 * i for i in range(30)]  # factors on amax/6 tried by the mse search (0.62 .. 1.20)
CALIB_IN = os.environ.get("A2E_CALIB_IN", "")
CALIB_OUT = os.environ.get("A2E_CALIB_OUT", "")
CACHE_DIR = os.environ.get("A2E_FP4_CACHE", "")
GPTQ_DAMP = float(os.environ.get("A2E_GPTQ_DAMP", "0.01"))
SO4 = os.environ.get("A2E_FP4_SO", F.SO4)
F.SO4 = SO4


# ------------------------------------------------------------------ rounding helpers
def e2m1_round(y):
    """Nearest E2M1 value (codes 0..15, sign in bit 3) and its float value, for y already divided by its scale."""
    mid = MID.to(y.device)
    mag = torch.bucketize(y.abs().clamp(max=6.0), mid)
    val = E2M1.to(y.device)[mag] * torch.where(y < 0, -1.0, 1.0)
    code = mag.to(torch.uint8) | ((y < 0).to(torch.uint8) << 3)
    return code, val


def e4m3(x):
    return x.clamp(min=2.0**-9, max=448.0).to(torch.float8_e4m3fn)


def pack(code, N, K):
    code = code.reshape(N, K)
    return code[:, 0::2] | (code[:, 1::2] << 4)


def block_scale_search(xb, g, d=None):
    """xb [R, B, 16] weights, g [R] row scale, d [B, 16] or None (importance) -> best E4M3 block scale [R, B]."""
    amax = xb.abs().amax(dim=2)
    best_err, best_b = None, None
    for f in SCALE_GRID:
        b = e4m3(amax / 6 / g[:, None] * f)
        s = b.float() * g[:, None]
        _, v = e2m1_round(xb / s[:, :, None])
        e = (v * s[:, :, None] - xb) ** 2
        if d is not None:
            e = e * d
        e = e.sum(dim=2)
        if best_err is None:
            best_err, best_b = e, b
        else:
            m = e < best_err
            best_err = torch.where(m, e, best_err)
            best_b = torch.where(m, b, best_b)
    return best_b


def quant_rtn(w, H=None, chunk=4096):
    return F.quantize_nvfp4(w, chunk)


def quant_mse(w, H=None, chunk=2048):
    w = w.detach()
    N, K = w.shape
    d = None if H is None else H.diagonal().float().clamp(min=1e-8).view(K // 16, 16).to(w.device)
    qw = torch.empty(N, K // 2, dtype=torch.uint8, device=w.device)
    bs = torch.empty(N, K // 16, dtype=torch.uint8, device=w.device)
    gs = torch.empty(N, dtype=torch.float32, device=w.device)
    for i in range(0, N, chunk):
        x = w[i : i + chunk].float()
        g = (x.abs().amax(dim=1) / (6 * 448)).clamp(min=1e-12)
        xb = x.view(x.shape[0], K // 16, 16)
        b = block_scale_search(xb, g, d)
        s = b.float() * g[:, None]
        code, _ = e2m1_round(xb / s[:, :, None])
        qw[i : i + chunk] = pack(code, x.shape[0], K)
        bs[i : i + chunk] = b.view(torch.uint8)
        gs[i : i + chunk] = g
    return qw, bs, gs


def quant_gptq(w, H, search=False, blocksize=128, damp=None):
    """GPTQ for the NVFP4-like format. w [N, K] (any float dtype), H [K, K] FP32 (sum of x x^T)."""
    damp = GPTQ_DAMP if damp is None else damp
    W = w.detach().float().clone()
    N, K = W.shape
    H = H.to(W.device, torch.float32).clone()
    dead = H.diagonal() == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    g = (W.abs().amax(dim=1) / (6 * 448)).clamp(min=1e-12)
    H.diagonal().add_(damp * H.diagonal().mean())
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    dH = H.diagonal()
    codes = torch.empty(N, K, dtype=torch.uint8, device=W.device)
    bsc = torch.empty(N, K // 16, dtype=torch.float8_e4m3fn, device=W.device)
    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        s = None
        for i in range(i2 - i1):
            col = i1 + i
            if col % 16 == 0:
                blk = W1[:, i : i + 16]
                if search:
                    b = block_scale_search(blk.unsqueeze(1), g, dH[col : col + 16].view(1, 16)).squeeze(1)
                else:
                    b = e4m3(blk.abs().amax(dim=1) / 6 / g)
                bsc[:, col // 16] = b
                s = b.float() * g
            wcol = W1[:, i]
            code, v = e2m1_round(wcol / s)
            q = v * s
            codes[:, col] = code
            err = (wcol - q) / Hinv1[i, i]
            W1[:, i:] -= err[:, None] * Hinv1[i, i:][None, :]
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return pack(codes, N, K), bsc.view(torch.uint8), g


# ------------------------------------------------------------------ calibration statistics
class CalibStats:
    """H = sum x x^T for each projection input, accumulated in place (works inside CUDA graph replays)."""

    def __init__(self, n_layers, hidden, inter, device):
        f = dict(device=device, dtype=torch.float32)
        self.H = {}
        for l in range(n_layers):
            for name, k in (("qkv", hidden), ("o", hidden), ("fc1", hidden), ("fc2", inter)):
                self.H[f"{l}.{name}"] = torch.zeros(k, k, **f)
        self.count = torch.zeros((), **f)

    def add(self, key, x):
        xf = x.reshape(-1, x.shape[-1]).float()
        self.H[key].addmm_(xf.t(), xf)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"H": {k: v.cpu() for k, v in self.H.items()}, "count": float(self.count.item())}, path)
        print(f"[a2e_fp4q] calibration statistics saved: {path} (tokens counted {self.count.item():.0f})", flush=True)


class CalibStep(torch.nn.Module):
    """BF16 decode step identical to the upstream DecodeStep, recording H for every projection input."""

    def __init__(self, vlm, stats: CalibStats):
        super().__init__()
        self.text, self.lm_head, self.stats = vlm.model.language_model, vlm.lm_head, stats

    def forward(self, input_ids, rope_delta, seqlens, k_caches, v_caches):
        from alpamayo2_edge.accel.decode import apply_rotary_pos_emb, kvcache_attention

        text = self.text
        hidden = text.embed_tokens(input_ids)
        rows = hidden.shape[0]
        positions = (seqlens.to(torch.long) + rope_delta).view(1, rows, 1).expand(3, rows, 1)
        cos, sin = text.rotary_emb(hidden, positions)
        st = self.stats
        st.count.add_(rows)
        for li, layer in enumerate(text.layers):
            attn, mlp = layer.self_attn, layer.mlp
            x = layer.input_layernorm(hidden)
            st.add(f"{li}.qkv", x)
            q, k, v = attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)
            seq = hidden.shape[1]
            q = q.reshape(rows, seq, -1, attn.head_dim)
            k = k.reshape(rows, seq, -1, attn.head_dim)
            v = v.reshape(rows, seq, -1, attn.head_dim)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
            dt = k_caches[li].dtype
            out = kvcache_attention(q.to(dt).contiguous(), k_caches[li], v_caches[li], k.to(dt).contiguous(),
                                    v.to(dt).contiguous(), seqlens, attn.scaling, True)
            out = out.reshape(rows, seq, -1)
            st.add(f"{li}.o", out)
            hidden = hidden + attn.o_proj(out)
            x = layer.post_attention_layernorm(hidden)
            st.add(f"{li}.fc1", x)
            h = mlp.act_fn(mlp.fc1(x)) if hasattr(mlp, "act_fn") else mlp.activation_fn(mlp.fc1(x))
            st.add(f"{li}.fc2", h)
            hidden = hidden + mlp.fc2(h)
        logits = self.lm_head(text.norm(hidden))
        seqlens.add_(1)
        return logits


_STATS = None


def _v_calib(vlm, handle):
    global _STATS
    cfg = vlm.config.text_config
    _STATS = CalibStats(cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size, vlm.lm_head.weight.device)
    if CALIB_OUT:
        atexit.register(lambda: _STATS.save(CALIB_OUT))
    return CalibStep(vlm, _STATS)


# ------------------------------------------------------------------ quantized layers
_HCACHE = None


def calib_H():
    global _HCACHE
    if _HCACHE is None:
        if not CALIB_IN:
            raise RuntimeError("A2E_CALIB_IN is not set (calibration statistics are needed for this method)")
        _HCACHE = torch.load(CALIB_IN, map_location="cpu")
        print(f"[a2e_fp4q] calibration statistics: {CALIB_IN} tokens={_HCACHE['count']:.0f}", flush=True)
    return _HCACHE["H"]


def quantize(method, w, key):
    if method == "rtn":
        return quant_rtn(w)
    # "mseu" uses no importance weights; the head ("head") has no calibration statistics.
    H = calib_H()[key].to(w.device) if method != "mseu" else None
    if method == "mse":
        return quant_mse(w, H)
    if method == "mseu":  # mse without importance weights
        return quant_mse(w, None)
    base, _, damp = method.partition(":")  # e.g. "gptq:0.1" = GPTQ with damping 0.1 * mean(diag H)
    damp = float(damp) if damp else None
    if base == "gptq":
        return quant_gptq(w, H, search=False, damp=damp)
    if base == "gptqm":
        return quant_gptq(w, H, search=True, damp=damp)
    raise ValueError(method)


class P4Linear(torch.nn.Module):
    """Pre-quantized FP4 weights (same buffers as a2e_fp4.Q4Linear)."""

    def __init__(self, qw, bs, gs, act=0):
        super().__init__()
        self.register_buffer("w", qw, persistent=False)
        self.register_buffer("bs", bs, persistent=False)
        self.register_buffer("gs", gs, persistent=False)
        self.act = int(act)

    def forward(self, x):
        return torch.ops.a2e4.fp4_gemv(x.to(torch.bfloat16), self.w, self.bs, self.gs, self.act)

    def nbytes(self):
        return self.w.numel() + self.bs.numel() + self.gs.numel() * 4


class P4Layer(torch.nn.Module):
    def __init__(self, parts, nq, nkv):
        super().__init__()
        self.nq, self.nkv = nq, nkv
        self.qkv = P4Linear(*parts["qkv"])
        self.o = P4Linear(*parts["o"])
        self.fc1 = P4Linear(*parts["fc1"], act=1)
        self.fc2 = P4Linear(*parts["fc2"])


def quantized_body(vlm, method):
    """{layer: {proj: (qw, bs, gs)}} for the 28 text layers, cached on disk per method."""
    path = Path(CACHE_DIR) / f"body_{method}.pt" if CACHE_DIR else None
    if path is not None and path.exists():
        dev = vlm.lm_head.weight.device
        data = torch.load(path, map_location=dev)
        print(f"[a2e_fp4q] loaded cached FP4 body: {path}", flush=True)
        return data
    out = {}
    with torch.no_grad():
        for li, layer in enumerate(vlm.model.language_model.layers):
            a, m = layer.self_attn, layer.mlp
            ws = {"qkv": torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], dim=0),
                  "o": a.o_proj.weight, "fc1": m.fc1.weight, "fc2": m.fc2.weight}
            out[li] = {k: quantize(method, w, f"{li}.{k}") for k, w in ws.items()}
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, path)
        print(f"[a2e_fp4q] saved FP4 body: {path}", flush=True)
    return out


def build_step(vlm, method="rtn", head="bf16"):
    F.load_ops()
    Q.load_ops()
    body = quantized_body(vlm, method)
    with torch.no_grad():
        layers = torch.nn.ModuleList(
            P4Layer(body[li], l.self_attn.q_proj.out_features, l.self_attn.k_proj.out_features)
            for li, l in enumerate(vlm.model.language_model.layers))
        if head == "fp8":
            qhead = Q.QLinear(vlm.lm_head.weight, 1)
        elif head == "fp4":
            hm = "mseu" if method != "rtn" else "rtn"
            qhead = P4Linear(*quantize(hm, vlm.lm_head.weight, "head"))
        else:
            qhead = None
    return Q.DecodeStepQ(vlm, layers, qhead)


def install(handle, method="rtn", head="bf16"):
    """Replace the official DecodeAccelerator.compiled_step (before any decode graph is captured)."""
    dec = Q.decode_accelerator(handle)
    if dec.step_graphs or dec.sampler.graphs:
        raise RuntimeError("install before the first decode call")
    step = build_step(dec.vlm, method, head)
    dec.compiled_step = Q.compiled(step, handle)
    body = sum(m.nbytes() for m in step.qlayers.modules() if isinstance(m, P4Linear))
    hb = step.qhead.nbytes() if step.qhead is not None else vlm_head_bytes(dec.vlm)
    return {"method": method, "head": head, "so4": SO4, "body_bytes": body, "head_bytes": hb}


def vlm_head_bytes(vlm):
    return vlm.lm_head.weight.numel() * vlm.lm_head.weight.element_size()


def _v(method, head):
    def make(vlm, handle):
        return Q.compiled(build_step(vlm, method, head), handle)
    return make


class _Variants(dict):
    """Variant names: fp4<method>[_d<damp in %>][_h8|_h4], e.g. fp4gptq_d10_h8 = GPTQ (damp 0.10), FP8 head."""

    def __missing__(self, name):
        if not name.startswith("fp4"):
            raise KeyError(name)
        parts = name[3:].split("_")
        method, head = parts[0], "bf16"
        for p in parts[1:]:
            if p.startswith("d"):
                method += ":" + str(int(p[1:]) / 100)
            elif p in ("h8", "h4"):
                head = {"h8": "fp8", "h4": "fp4"}[p]
            else:
                raise KeyError(name)
        if method.split(":")[0] not in ("rtn", "mse", "mseu", "gptq", "gptqm"):
            raise KeyError(name)
        return _v(method, head)

    def __contains__(self, name):
        try:
            self[name]
            return True
        except KeyError:
            return False


VARIANTS = _Variants(calib=_v_calib)
# wide_eval.py copies the dict with update(), so list the names we use explicitly as well.
for _m in ("rtn", "mse", "mseu", "gptq", "gptqm"):
    for _d in ([""] if not _m.startswith("gptq") else ["", "_d1", "_d5", "_d10", "_d20", "_d50", "_d100"]):
        for _h in ("", "_h8", "_h4"):
            _n = f"fp4{_m}{_d}{_h}"
            VARIANTS[_n] = VARIANTS[_n]

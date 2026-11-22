#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
"""Minimal TensorRT networks around Edge-LLM's Nvfp4A16GemmPlugin, to find which graph context breaks it.

Weights: synthetic random NVFP4 (ModelOpt layout) or the real checkpoint (--ckpt, layer 0 k_proj/v_proj), packed by
the builder's pack_nvfp4_a16_linear and fed as network constants (like an embedded-weight engine).
Variants (--variant):
  plain     : x(input) -> plugin -> out
  pre       : x(input) -> x*g (elementwise, g const) -> plugin -> out        (plugin input is an intermediate)
  pre_post  : x -> x*g -> plugin_k, plugin_v -> concat(k, v) -> out
  rms       : x -> RMSNorm (explicit ops) -> plugin_k, plugin_v -> concat -> out
  qkv       : x -> RMSNorm -> plugin_q, plugin_k, plugin_v -> concat(q, concat(k, v)) -> *1 -> out
  qkv_flat  : same with concat(q, k, v)
  qkv_plug  : x -> RMSNorm -> plugin_q, plugin_k, plugin_v -> concat(q, concat(k, v)) -> plugin_c (K=4096) -> out
              (the concat feeds another plugin, like AttentionPlugin in the real engine)
  qkv_plug_flat : same with concat(q, k, v)
  plug2     : x -> plugin_k, plugin_v -> concat(k, v) -> plugin_c (K=2048) -> out   (smallest form)
--copy-fix 1 inserts an elementwise x*1 after each projection plugin (workaround candidate; optimized away),
           2 casts each projection output to FP32, concatenates in FP32 and casts the concat back to FP16,
           3 casts each projection output FP16 -> FP32 -> FP16 before the concat.
--profiles 2 adds a second optimization profile (M = 1) like Edge-LLM's prefill / decode profiles.
Each runs with M in --ms, and reports rel. error vs FP32 dequantized reference.
Env __LUNOWUD can be set by --lunowud to mimic Edge-LLM's builder flags.
"""
import argparse, os

import numpy as np
import tensorrt as trt
import torch

FP4 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)


def rand_weight(n, k, g):
    codes = torch.randint(0, 16, (n, k), generator=g, dtype=torch.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4))
    sc = (torch.rand((n, k // 16), generator=g) * 400 + 8).to(torch.float8_e4m3fn)
    return packed, sc, 2.0e-4


def ckpt_weight(path, name):
    from safetensors import safe_open
    f = safe_open(path, "pt")
    return f.get_tensor(name + ".weight"), f.get_tensor(name + ".weight_scale"), float(f.get_tensor(name + ".weight_scale_2"))


def dequant(packed, sc, gs):
    n = packed.shape[0]
    codes = torch.empty((n, packed.shape[1] * 2), dtype=torch.uint8)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = packed >> 4
    mag = FP4[(codes & 7).long()]
    v = torch.where((codes & 8) != 0, -mag, mag)
    return v * sc.float().repeat_interleave(16, 1) * gs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin-lib", required=True)
    ap.add_argument("--variant", default="plain")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--ms", default="1,33,3261")
    ap.add_argument("--max-m", type=int, default=4096)
    ap.add_argument("--attr-max-m", type=int, default=0, help="plugin max_m attribute (Edge-LLM uses 0 = auto)")
    ap.add_argument("--lunowud", default=None)
    ap.add_argument("--strongly-typed", type=int, default=1)
    ap.add_argument("--profiles", type=int, default=1)
    ap.add_argument("--copy-fix", type=int, default=0)
    ap.add_argument("--profile-index", type=int, default=0)
    a = ap.parse_args()
    if a.lunowud is not None:
        os.environ["__LUNOWUD"] = a.lunowud
    from tensorrt_edgellm.builder.weight_packing import nvfp4 as P
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    import ctypes
    ctypes.CDLL(a.plugin_lib, mode=ctypes.RTLD_GLOBAL)
    K = 2048
    g = torch.Generator().manual_seed(0)
    if a.ckpt:
        ws = {"c4096": rand_weight(256, 4096, g), "c2048": rand_weight(256, 2048, g),
              "q": ckpt_weight(a.ckpt, "model.layers.0.self_attn.q_proj"),
              "k": ckpt_weight(a.ckpt, "model.layers.0.self_attn.k_proj"),
              "v": ckpt_weight(a.ckpt, "model.layers.0.self_attn.v_proj")}
    else:
        ws = {"q": rand_weight(2048, K, g), "k": rand_weight(1024, K, g), "v": rand_weight(1024, K, g)}
    ws["c4096"] = rand_weight(256, 4096, g)
    ws["c2048"] = rand_weight(256, 2048, g)
    builder = trt.Builder(logger)
    flags = (1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)) if a.strongly_typed else 0
    net = builder.create_network(flags)
    cfg = builder.create_builder_config()
    x = net.add_input("x", trt.float16, (-1, K))
    keep = []

    def const(arr, dtype=None):
        arr = np.ascontiguousarray(arr)
        keep.append(arr)
        return net.add_constant(arr.shape, trt.Weights(arr)).get_output(0)

    gamma = (torch.rand(K, generator=g) + 0.5).half()
    creator = trt.get_plugin_registry().get_creator("Nvfp4A16GemmPlugin", "1", "")

    def plugin(inp, key):
        packed, sc, gs = ws[key]
        kk = packed.shape[1] * 2
        q, s, ps, nl, npad = P.pack_nvfp4_a16_linear(packed.numpy(), sc.view(torch.uint8).numpy(), gs)
        qt = const(np.ascontiguousarray(q).view(np.int8).reshape(1, kk // 16, 8 * npad))
        st = const(np.ascontiguousarray(s).view(np.int8).reshape(1, kk // 16, npad))
        gt = const(np.ascontiguousarray(ps).astype(np.float16))
        fc = trt.PluginFieldCollection([trt.PluginField("gemm_n", np.array([npad], np.int32), trt.PluginFieldType.INT32),
                                        trt.PluginField("gemm_k", np.array([kk], np.int32), trt.PluginFieldType.INT32),
                                        trt.PluginField("max_m", np.array([a.attr_max_m], np.int32), trt.PluginFieldType.INT32)])
        p = creator.create_plugin("a16_" + key, fc, trt.TensorRTPhase.BUILD)
        o = net.add_plugin_v3([inp, qt, st, gt], [], p).get_output(0)
        if a.copy_fix == 1 and not key.startswith("c"):
            o = net.add_elementwise(o, const(np.ones((1, 1), np.float16)), trt.ElementWiseOperation.PROD).get_output(0)
        elif a.copy_fix in (2, 3) and not key.startswith("c"):
            o = net.add_cast(o, trt.float32).get_output(0)
            if a.copy_fix == 3:
                o = net.add_cast(o, trt.float16).get_output(0)
        return o

    def cat(ts):
        c = net.add_concatenation(ts); c.axis = 1
        o = c.get_output(0)
        if a.copy_fix == 2:
            o = net.add_cast(o, trt.float16).get_output(0)
        return o

    v = a.variant
    if v in ("plain", "plug2"):
        h = x
    elif v in ("pre", "pre_post"):
        h = net.add_elementwise(x, const(gamma.numpy().reshape(1, K)), trt.ElementWiseOperation.PROD).get_output(0)
    elif v in ("rms", "qkv", "qkv_flat", "qkv_plug", "qkv_plug_flat"):
        sq = net.add_elementwise(x, x, trt.ElementWiseOperation.PROD).get_output(0)
        h = net.add_normalization(x, const(gamma.numpy().reshape(1, K)), const(np.zeros((1, K), np.float16)), 1 << 1)
        h.epsilon = 1e-5
        try:
            h.compute_precision = trt.float32
        except Exception:
            pass
        # INormalization is LayerNorm; RMSNorm in Edge-LLM is a plugin/explicit ops. Use explicit RMS instead:
        mean = net.add_reduce(sq, trt.ReduceOperation.AVG, 1 << 1, True).get_output(0)
        eps = const(np.array([[1e-5]], np.float16))
        rs = net.add_unary(net.add_elementwise(mean, eps, trt.ElementWiseOperation.SUM).get_output(0),
                           trt.UnaryOperation.SQRT).get_output(0)
        hn = net.add_elementwise(x, rs, trt.ElementWiseOperation.DIV).get_output(0)
        h = net.add_elementwise(hn, const(gamma.numpy().reshape(1, K)), trt.ElementWiseOperation.PROD).get_output(0)
    else:
        raise SystemExit(v)
    if v in ("plain", "pre"):
        out = plugin(h, "k")
    elif v in ("qkv_plug", "qkv_plug_flat"):
        pq, pk, pv = plugin(h, "q"), plugin(h, "k"), plugin(h, "v")
        if v == "qkv_plug":
            c = cat([pq, cat([pk, pv])]) if a.copy_fix != 2 else None
            if a.copy_fix == 2:  # keep FP32 through both concats
                kv = net.add_concatenation([pk, pv]); kv.axis = 1
                c0 = net.add_concatenation([pq, kv.get_output(0)]); c0.axis = 1
                c = net.add_cast(c0.get_output(0), trt.float16).get_output(0)
        else:
            c = cat([pq, pk, pv])
        out = plugin(c, "c4096")
    elif v == "plug2":
        out = plugin(cat([plugin(h, "k"), plugin(h, "v")]), "c2048")
    elif v in ("qkv", "qkv_flat"):
        pq, pk, pv = plugin(h, "q"), plugin(h, "k"), plugin(h, "v")
        if v == "qkv":
            kv = net.add_concatenation([pk, pv]); kv.axis = 1
            c = net.add_concatenation([pq, kv.get_output(0)]); c.axis = 1
        else:
            c = net.add_concatenation([pq, pk, pv]); c.axis = 1
        out = net.add_elementwise(c.get_output(0), const(np.ones((1, 4096), np.float16)),
                                  trt.ElementWiseOperation.PROD).get_output(0)
    else:
        out = net.add_concatenation([plugin(h, "k"), plugin(h, "v")])
        out.axis = 1
        out = out.get_output(0)
    out.name = "out"
    net.mark_output(out)
    prof = builder.create_optimization_profile()
    prof.set_shape("x", (1, K), (64, K), (a.max_m, K))
    cfg.add_optimization_profile(prof)
    if a.profiles == 2:
        p2 = builder.create_optimization_profile()
        p2.set_shape("x", (1, K), (1, K), (1, K))
        cfg.add_optimization_profile(p2)
    ser = builder.build_serialized_network(net, cfg)
    eng = trt.Runtime(logger).deserialize_cuda_engine(ser)
    ctx = eng.create_execution_context()
    Wk = dequant(*ws["k"]).cuda(); Wv = dequant(*ws["v"]).cuda(); Wq = dequant(*ws["q"]).cuda()
    Wc4 = dequant(*ws["c4096"]).cuda(); Wc2 = dequant(*ws["c2048"]).cuda()
    st = torch.cuda.Stream()
    if a.profile_index:
        ctx.set_optimization_profile_async(a.profile_index, st.cuda_stream)
    for m in [int(s) for s in a.ms.split(",")]:
        xin = (torch.randn((m, K), generator=g) * 1.0).half().cuda()
        xf = xin.float()
        if v in ("pre", "pre_post"):
            hf = (xin * gamma.cuda()).float()
        elif v in ("rms", "qkv", "qkv_flat", "qkv_plug", "qkv_plug_flat"):
            hf = (xf / torch.sqrt((xf * xf).mean(1, keepdim=True) + 1e-5) * gamma.cuda().float()).half().float()
        else:
            hf = xf
        if v in ("plain", "pre"):
            ref = hf @ Wk.t()
        elif v in ("qkv", "qkv_flat"):
            ref = torch.cat([hf @ Wq.t(), hf @ Wk.t(), hf @ Wv.t()], 1)
        elif v in ("qkv_plug", "qkv_plug_flat"):
            ref = torch.cat([hf @ Wq.t(), hf @ Wk.t(), hf @ Wv.t()], 1).half().float() @ Wc4.t()
        elif v == "plug2":
            ref = torch.cat([hf @ Wk.t(), hf @ Wv.t()], 1).half().float() @ Wc2.t()
        else:
            ref = torch.cat([hf @ Wk.t(), hf @ Wv.t()], 1)
        o = torch.full(ref.shape, float("nan"), dtype=torch.float16, device="cuda")
        ctx.set_input_shape("x", tuple(xin.shape))
        ctx.set_tensor_address("x", xin.data_ptr()); ctx.set_tensor_address("out", o.data_ptr())
        ctx.execute_async_v3(st.cuda_stream); st.synchronize()
        rel = ((o.float() - ref).norm() / ref.norm()).item()
        cos = torch.nn.functional.cosine_similarity(o.float().flatten(), ref.flatten(), dim=0).item()
        print(f"variant={v} copy_fix={a.copy_fix} profiles={a.profiles} ckpt={bool(a.ckpt)} lunowud={a.lunowud!r} M={m}: rel {rel:.3e} cos {cos:.5f} "
              f"nan={torch.isnan(o).any().item()}", flush=True)


if __name__ == "__main__":
    main()

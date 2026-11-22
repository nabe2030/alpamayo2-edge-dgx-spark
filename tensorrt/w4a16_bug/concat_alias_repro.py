#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
"""Pure TensorRT reproducer (no Edge-LLM): two plugin outputs -> axis-1 concat -> plugin consumer.

The plugins are trivial Python plugins (tensorrt.plugin, torch implementation):
  a2e::copy(x) : y = x,   a2e::double(x) : y = 2x   (same shape, no attributes)
Graph:  x[M, K] -> copy(x) -> a ; double(x) -> b ; c = concat([a, b], axis=1) [M, 2K] ; out = copy(c)
Expected: out = [x, 2x]. Reports the max error for several M.
--fix fp32 : concatenate in FP32 and cast back (workaround), --fix none : as above,
--fix identity / max / shuffle : other copy candidates (IIdentityLayer, max(x, x), reshape).
"""
import argparse
from typing import Tuple

import numpy as np
import tensorrt as trt
import tensorrt.plugin as trtp
import torch


def _desc(inp0: trtp.TensorDesc) -> trtp.TensorDesc:
    return inp0.like()


def _run(inp0, outputs, stream, f):
    with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
        x = torch.as_tensor(inp0, device="cuda")
        y = torch.as_tensor(outputs[0], device="cuda")
        y.copy_(x * f)


@trtp.register("a2e::copy")
def copy_desc(inp0: trtp.TensorDesc) -> trtp.TensorDesc:
    return inp0.like()


@trtp.impl("a2e::copy")
def copy_impl(inp0: trtp.Tensor, outputs: Tuple[trtp.Tensor], stream: int) -> None:
    _run(inp0, outputs, stream, 1.0)


@trtp.register("a2e::double")
def double_desc(inp0: trtp.TensorDesc) -> trtp.TensorDesc:
    return inp0.like()


@trtp.impl("a2e::double")
def double_impl(inp0: trtp.Tensor, outputs: Tuple[trtp.Tensor], stream: int) -> None:
    _run(inp0, outputs, stream, 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--ms", default="1,2,33,512")
    ap.add_argument("--fix", default="none", choices=["none", "fp32", "identity", "max", "shuffle"])
    ap.add_argument("--consumer", default="plugin", choices=["plugin", "output"])
    ap.add_argument("--strongly-typed", type=int, default=1)
    a = ap.parse_args()
    K = a.K
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network((1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)) if a.strongly_typed else 0)
    x = net.add_input("x", trt.float16, (-1, K))
    pa = net.add_plugin(trtp.op.a2e.copy(x)).get_output(0)
    pb = net.add_plugin(trtp.op.a2e.double(x)).get_output(0)
    if a.fix == "fp32":
        pa = net.add_cast(pa, trt.float32).get_output(0)
        pb = net.add_cast(pb, trt.float32).get_output(0)
    if a.fix == "identity":
        pa = net.add_identity(pa).get_output(0); pb = net.add_identity(pb).get_output(0)
    elif a.fix == "max":
        pa = net.add_elementwise(pa, pa, trt.ElementWiseOperation.MAX).get_output(0)
        pb = net.add_elementwise(pb, pb, trt.ElementWiseOperation.MAX).get_output(0)
    elif a.fix == "shuffle":
        sa = net.add_shuffle(pa); sa.reshape_dims = (0, K); pa = sa.get_output(0)
        sb = net.add_shuffle(pb); sb.reshape_dims = (0, K); pb = sb.get_output(0)
    c = net.add_concatenation([pa, pb]); c.axis = 1
    c = c.get_output(0)
    if a.fix == "fp32":
        c = net.add_cast(c, trt.float16).get_output(0)
    out = net.add_plugin(trtp.op.a2e.copy(c)).get_output(0) if a.consumer == "plugin" else c
    out.name = "out"
    net.mark_output(out)
    cfg = builder.create_builder_config()
    prof = builder.create_optimization_profile()
    prof.set_shape("x", (1, K), (64, K), (4096, K))
    cfg.add_optimization_profile(prof)
    eng = trt.Runtime(logger).deserialize_cuda_engine(builder.build_serialized_network(net, cfg))
    ctx = eng.create_execution_context()
    for m in [int(s) for s in a.ms.split(",")]:
        xin = torch.randn((m, K), device="cuda").half()
        o = torch.full((m, 2 * K), float("nan"), device="cuda", dtype=torch.float16)
        ctx.set_input_shape("x", (m, K))
        ctx.set_tensor_address("x", xin.data_ptr()); ctx.set_tensor_address("out", o.data_ptr())
        st = torch.cuda.Stream()
        ctx.execute_async_v3(st.cuda_stream); st.synchronize()
        ref = torch.cat([xin, 2 * xin], 1)
        err = (o.float() - ref.float()).abs().max().item()
        print(f"TRT {trt.__version__} {torch.cuda.get_device_name()} fix={a.fix} consumer={a.consumer} strongly_typed={a.strongly_typed} M={m}: "
              f"max abs err {err:.3e} {'OK' if err < 1e-2 else 'WRONG'}", flush=True)


if __name__ == "__main__":
    main()

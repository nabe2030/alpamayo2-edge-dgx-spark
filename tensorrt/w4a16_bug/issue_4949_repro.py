# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# issue_4949_repro.py — the pure-TensorRT reproducer attached to NVIDIA/TensorRT#4949
#   (https://github.com/NVIDIA/TensorRT/issues/4949). The code below these comments is the script in the issue, unchanged.
#   Graph: x[M, K] -> plugin copy(x) -> a ; x -> plugin double(x) -> b ; c = concat([a, b], axis=1) ; out = plugin copy(c).
#   Expected out == [x, 2x]. On DGX Spark (GB10) with the tensorrt-cu13 wheels 10.15.1.29, 10.16.1.11 and 11.5.0.63,
#   M = 1 is correct and M >= 2 is wrong: TensorRT assigns the plugin outputs directly into the concat buffer without a copy.
#   Needs: tensorrt (pip wheel, includes tensorrt.plugin), torch with CUDA. Run: python issue_4949_repro.py
#   Our output on 10.16.1.11: M=1 max abs err 0 (OK); M=2 6.8, M=33 9.7, M=512 9.4 (WRONG).
from typing import Tuple

import tensorrt as trt
import tensorrt.plugin as trtp
import torch


def _run(inp0, outputs, stream, factor):
    with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
        x = torch.as_tensor(inp0, device="cuda")
        y = torch.as_tensor(outputs[0], device="cuda")
        y.copy_(x * factor)


@trtp.register("repro::copy")
def copy_desc(inp0: trtp.TensorDesc) -> trtp.TensorDesc:
    return inp0.like()


@trtp.impl("repro::copy")
def copy_impl(inp0: trtp.Tensor, outputs: Tuple[trtp.Tensor], stream: int) -> None:
    _run(inp0, outputs, stream, 1.0)


@trtp.register("repro::double")
def double_desc(inp0: trtp.TensorDesc) -> trtp.TensorDesc:
    return inp0.like()


@trtp.impl("repro::double")
def double_impl(inp0: trtp.Tensor, outputs: Tuple[trtp.Tensor], stream: int) -> None:
    _run(inp0, outputs, stream, 2.0)


K = 256
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
x = net.add_input("x", trt.float16, (-1, K))
a = net.add_plugin(trtp.op.repro.copy(x)).get_output(0)
b = net.add_plugin(trtp.op.repro.double(x)).get_output(0)
cat = net.add_concatenation([a, b])
cat.axis = 1
out = net.add_plugin(trtp.op.repro.copy(cat.get_output(0))).get_output(0)
out.name = "out"
net.mark_output(out)

cfg = builder.create_builder_config()
prof = builder.create_optimization_profile()
prof.set_shape("x", (1, K), (64, K), (4096, K))
cfg.add_optimization_profile(prof)
engine = trt.Runtime(logger).deserialize_cuda_engine(builder.build_serialized_network(net, cfg))
ctx = engine.create_execution_context()

for m in (1, 2, 33, 512):
    xin = torch.randn((m, K), device="cuda").half()
    o = torch.full((m, 2 * K), float("nan"), device="cuda", dtype=torch.float16)
    ctx.set_input_shape("x", (m, K))
    ctx.set_tensor_address("x", xin.data_ptr())
    ctx.set_tensor_address("out", o.data_ptr())
    s = torch.cuda.Stream()
    ctx.execute_async_v3(s.cuda_stream)
    s.synchronize()
    ref = torch.cat([xin, 2 * xin], 1)
    err = (o.float() - ref.float()).abs().max().item()
    print(f"TRT {trt.__version__} M={m}: max abs err {err:.3e} {'OK' if err < 1e-2 else 'WRONG'}")

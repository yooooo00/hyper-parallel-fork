# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""CP acceptance retaining the default real NPU fused attention interface.

Uses the predeclared GQA parallel tolerances: BF16 rtol=.03/atol=.003,
MXFP8 NRMSE .10 and maximum absolute error .5 times the reference maximum.
"""

import copy
from functools import partial
import json
import os

import pytest
import torch
import torch.distributed as dist

from hyper_parallel.components.functional import npu_fusion_attention_forward
from hyper_parallel.components.quantization.quantizers.mxfp8 import MXFP8Quantizer
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.apply import apply_sharding_plan
from hyper_parallel.models.replacement import _apply_module_replacement_actions
from hyper_parallel.trainer.config import entries_to_module_replacements
from hyper_parallel.trainer.config.resolver import resolve_config
from tests.common.gqa_family import attention_inputs, replace_attention
from tests.torch.attention._test_gqa_parallel import (
    _check, _check_weights, _config, _inputs_for_rank, _parameter_gradients, _partitioned_reference,
)
from tests.torch.attention._test_mxfp8_gqa_attention import CountingLinearOps, _device


def _check_default_kernel(kind, low, mesh, device, backend, *, quantized_reference=False):
    """Compare the production CP path with a local, explicitly masked oracle."""
    config_dict = _config(kind, low, 1, 2, False).to_dict()
    del config_dict["plan_overrides"][0]["replace_module"]["attention_interface"]
    config = resolve_config(config_dict)
    model = config.model.build().to(device)
    reference, _ = replace_attention(
        copy.deepcopy(model.self_attn), use_mxfp8=quantized_reference,
        attention_interface=partial(npu_fusion_attention_forward, next_tokens=1048576),
    )
    _apply_module_replacement_actions(model, entries_to_module_replacements(config.plan_overrides))
    assert model.self_attn.attention_interface is npu_fusion_attention_forward
    plan = ShardingPlanner().plan(model, mesh, cp_size=2, sequence_parallel=False)
    layouts = {f"{fqn}.{name}".removeprefix("self_attn."): placements
               for fqn, spec in plan.modules.items() for name, placements in spec.params.items()}
    model, _ = apply_sharding_plan(model, plan, mesh)
    _check_weights(model, reference, layouts, mesh, 1)
    assert model.self_attn._hp_attention_cp_original_interface is npu_fusion_attention_forward
    counter = CountingLinearOps(backend)
    if low:
        for projection in (model.self_attn.linear_qkv, model.self_attn.o_proj):
            projection.quantizer = MXFP8Quantizer(npu_ops=counter)
    # Non-tile-aligned query lengths exposed an incorrect additional local
    # causal window. The reference explicitly leaves the global mask in charge.
    inputs = attention_inputs(sequence=96, device=str(device))
    inputs["hidden_states"] = inputs["hidden_states"][:1]
    inputs["position_embeddings"] = tuple(value[:1] for value in inputs["position_embeddings"])
    full_hidden = inputs["hidden_states"].detach().clone().requires_grad_()
    expected, replicas = _partitioned_reference(reference, {**inputs, "hidden_states": full_hidden}, 2)
    upstream = torch.randn_like(expected)
    (expected * upstream).sum().backward()
    local_inputs = _inputs_for_rank(inputs, mesh, 2)
    hidden = local_inputs["hidden_states"].detach().clone().requires_grad_()
    actual = model(**{**local_inputs, "hidden_states": hidden})
    local_rank = mesh["cp"].get_local_rank()
    (actual * upstream.chunk(2, dim=1)[local_rank]).sum().backward()
    compare_low = low and not quantized_reference
    _check("npu_cp/output", actual, expected.chunk(2, dim=1)[local_rank], compare_low)
    _check("npu_cp/input_gradient", hidden.grad, full_hidden.grad.chunk(2, dim=1)[local_rank], compare_low)
    _parameter_gradients(model, replicas, layouts, mesh, (1, 2), compare_low)
    assert counter.calls == (6 if low else 0)
    print(json.dumps({"kind": kind, "mxfp8": low, "rank": dist.get_rank(),
                      "reference_mxfp8": quantized_reference,
                      "attention_interface": "npu_fusion_attention_forward",
                      "fp8_calls": counter.calls}), flush=True)


@pytest.mark.parametrize("kind", ["llama", "qwen3_moe"])
@pytest.mark.parametrize("low", [False, True])
def test_default_npu_cp(kind: str, low: bool) -> None:
    """Keep the Family-selected CP wrapper and default real attention kernel.

    Args:
        kind: Native attention protocol: Llama or Qwen3-MoE.
        low: Whether the Family selects MXFP8 projections.
    """
    device, backend = _device()
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("Run this case with torchrun --nproc-per-node=2")
    dist.init_process_group("hccl")
    try:
        torch.manual_seed(793)
        torch.npu.manual_seed(793)
        mesh = init_device_mesh("npu", (2,), mesh_dim_names=("cp",))
        _check_default_kernel(kind, low, mesh, device, backend)
        torch.npu.synchronize()
    finally:
        dist.destroy_process_group()


def test_default_npu_cp_quantized_reference() -> None:
    """Isolate CP correctness from the separately checked BF16 quantization error."""
    device, backend = _device()
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("Run this case with torchrun --nproc-per-node=2")
    dist.init_process_group("hccl")
    try:
        torch.manual_seed(793)
        torch.npu.manual_seed(793)
        mesh = init_device_mesh("npu", (2,), mesh_dim_names=("cp",))
        _check_default_kernel("qwen3_moe", True, mesh, device, backend, quantized_reference=True)
        torch.npu.synchronize()
    finally:
        dist.destroy_process_group()

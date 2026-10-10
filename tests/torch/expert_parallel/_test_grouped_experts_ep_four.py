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
"""Four-rank Experts Family acceptance with dispatch-order matched references.

Every numerical comparison requires exact equality: reference and distributed
model use the same local expert token order, quantization blocks and reduction.
The tested model always receives its local expert tensors from planner/apply.
"""

import copy
import json
import os

import pytest
import torch
import torch.distributed as dist

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.distributed._builder.applier import _expert_mesh_layout
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.apply import apply_sharding_plan
from hyper_parallel.models.replacement import _apply_module_replacement_actions
from hyper_parallel.trainer.config import entries_to_module_replacements, entries_to_plan_overrides
from hyper_parallel.trainer.config.resolver import resolve_config
from tests.common.grouped_experts_family import TinyMoE, family_config
from tests.torch.expert_parallel._test_grouped_experts_mxfp8 import CountingMXOps, _device, _pack_physical


def _configuration(variant: str):
    config = family_config(use_mxfp8=variant != "baseline", use_fused_mxfp8=variant == "fused").to_dict()
    config["accelerator"] = {"ep_size": 4}
    config["plan_overrides"].append({
        "match": "mlp", "when": "ep", "region_dispatch": False,
        "local_compute_fn": {"_target_":
            "hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel.qwen3moe_ep_compute_fn",
            "use_grouped_gemm": True},
    })
    return resolve_config(config)


def _local_reference(experts, inputs, routes, rank, orders):
    local_experts = experts.num_experts // len(inputs)
    local = copy.deepcopy(experts)
    first = rank * local_experts
    for name in ("gate_up_proj", "down_proj"):
        tensor = getattr(experts, name)[first:first + local_experts].detach().clone()
        setattr(local, name, torch.nn.Parameter(tensor))
    parts, ids, sizes = [], [], []
    for source, route, order in zip(inputs, routes, orders):
        selected = order[(route.flatten()[order] // local_experts) == rank]
        tokens = selected // route.shape[-1]
        parts.append(source.reshape(-1, 128)[tokens])
        ids.append(route.flatten()[selected] - first)
        sizes.append(selected.numel())
    received, local_ids = torch.cat(parts), torch.cat(ids)
    permutation = local_ids.argsort()
    counts = torch.bincount(local_ids, minlength=local_experts)
    sorted_output = local.forward_expert_major(received[permutation], counts)
    output = torch.empty_like(sorted_output)
    output[permutation] = sorted_output
    return local, output.split(sizes)


def _ordered_reference(experts, inputs, routes, scores):
    local_experts = experts.num_experts // len(inputs)
    orders = [((route.flatten() // local_experts) * experts.num_experts + route.flatten()).argsort()
              for route in routes]
    locals_by_rank, returned = [], [[] for _ in inputs]
    for rank in range(len(inputs)):
        local, parts = _local_reference(experts, inputs, routes, rank, orders)
        locals_by_rank.append(local)
        for sender, part in enumerate(parts):
            returned[sender].append(part)
    outputs = []
    for source, route, score, order, parts in zip(inputs, routes, scores, orders, returned):
        values = torch.cat(parts) * score.flatten()[order, None].to(source.dtype)
        output = torch.zeros_like(source.reshape(-1, 128))
        output.index_add_(0, order // route.shape[-1], values)
        outputs.append(output.reshape_as(source))
    return locals_by_rank, outputs


def _case_inputs(device, empty_rank):
    torch.manual_seed(170 + int(empty_rank))
    inputs = [torch.randn(1, 35, 128, dtype=torch.bfloat16, device=device).requires_grad_() for _ in range(4)]
    active_experts = 3 if empty_rank else 4
    routes = [torch.tensor([[(token + sender) % active_experts, (token + sender + 1) % active_experts]
                            for token in range(35)], device=device) for sender in range(4)]
    scores = [torch.tensor([[.137, .863]] * 35, device=device).requires_grad_() for _ in range(4)]
    upstream = [torch.randn_like(value) for value in inputs]
    return inputs, routes, scores, upstream


def _check_case(model, source_experts, rank, device, empty_rank):
    inputs, routes, scores, upstream = _case_inputs(device, empty_rank)
    reference_experts, expected = _ordered_reference(source_experts, inputs, routes, scores)
    sum((output * gradient).sum() for output, gradient in zip(expected, upstream)).backward()
    actual_input = inputs[rank].detach().clone().requires_grad_()
    actual_scores = scores[rank].detach().clone().requires_grad_()
    model.mlp.gate.indices, model.mlp.gate.scores = routes[rank], actual_scores
    model.zero_grad(set_to_none=True)
    calls = []
    handle = model.mlp.experts.register_forward_hook(
        lambda _module, args, output: calls.append((args[0].shape[0], output.shape[0])))
    with torch.autograd.graph.saved_tensors_hooks(_pack_physical, lambda tensor: tensor):
        output = model(actual_input)
        (output * upstream[rank]).sum().backward()
    handle.remove()
    torch.testing.assert_close(output, expected[rank], rtol=0, atol=0)
    torch.testing.assert_close(actual_input.grad, inputs[rank].grad, rtol=0, atol=0)
    torch.testing.assert_close(actual_scores.grad, scores[rank].grad, rtol=0, atol=0)
    for name, parameter in model.mlp.experts.named_parameters():
        torch.testing.assert_close(parameter.grad, reference_experts[rank].get_parameter(name).grad, rtol=0, atol=0)
    assert len(calls) == 1
    if empty_rank and rank == 3:
        assert calls[0] == (0, 0)
        assert all(torch.count_nonzero(parameter.grad) == 0 for parameter in model.mlp.experts.parameters())
    print(json.dumps({"rank": rank, "empty_rank_case": empty_rank, "received_tokens": calls[0][0],
                      "output_dx_dw_routing_grad": "exact"}), flush=True)


@pytest.mark.parametrize("variant", ["baseline", "mxfp8", "fused"])
def test_ep_four(variant: str) -> None:
    """Derive EP=4 from the dense mesh and validate actual one-expert shards.

    Args:
        variant: Baseline, non-fused MXFP8 or fused expert implementation.
    """
    device, backend = _device()
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("Run this acceptance case with torchrun --nproc-per-node=4")
    dist.init_process_group("hccl")
    rank = dist.get_rank()
    try:
        torch.manual_seed(793)
        model = TinyMoE(dtype=torch.bfloat16).to(device)
        config = _configuration(variant)
        _apply_module_replacement_actions(model, entries_to_module_replacements(config.plan_overrides),
                                          context={"ep": True, "ep_size": 4})
        source_experts = copy.deepcopy(model.mlp.experts)
        mesh = init_device_mesh("npu", (4, 1), mesh_dim_names=("dp", "tp"))
        expert_shape, expert_names, ranks = _expert_mesh_layout(mesh, mesh.mesh_dim_names, 4)
        assert expert_shape == (1, 4) and ranks == (0, 1, 2, 3)
        overrides = entries_to_plan_overrides(config.plan_overrides, ep_size=4)
        plan = ShardingPlanner(plan_overrides=overrides).plan(model, mesh, ep_size=4, sequence_parallel=False)
        print(plan.explain(), flush=True)
        model, shard_info = apply_sharding_plan(model, plan, mesh)
        expert = model.mlp.experts
        assert expert.gate_up_proj.shape == (1, 128, 512)
        assert expert.down_proj.shape == (1, 256, 128)
        assert expert.local_expert_count == 1 and expert.ep_use_grouped_gemm
        for name in ("gate_up_proj", "down_proj"):
            assert f"mlp.experts.{name}" in shard_info
            torch.testing.assert_close(expert.get_parameter(name), source_experts.get_parameter(name)[rank:rank + 1],
                                       rtol=0, atol=0)
        counter = CountingMXOps(backend)
        if variant != "baseline":
            expert.quantizer._npu_ops = counter
        for empty_rank in (False, True):
            _check_case(model, source_experts, rank, device, empty_rank)
        if variant != "baseline":
            assert counter.calls
        if variant == "fused":
            assert counter.fused_forward and counter.fused_backward
        print(json.dumps({"rank": rank, "variant": variant, "expert_mesh": expert_shape,
                          "expert_axes": expert_names, "expert_ranks": ranks, "local_experts": 1,
                          "fp8_gmm_calls": len(counter.calls),
                          "fused_calls": [counter.fused_forward, counter.fused_backward]}), flush=True)
        torch.npu.synchronize()
    finally:
        dist.destroy_process_group()

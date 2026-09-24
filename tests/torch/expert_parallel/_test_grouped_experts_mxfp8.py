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
"""Real A5 MXFP8 family acceptance: single-device lifecycle and planner-owned EP=2.

Thresholds fixed before execution: BF16 vs MXFP8 NRMSE <= .10 and
max_abs <= .5 * reference_abs_max + 1e-6 (output/dX/dW separately).
Repeated backward and EP vs order-matched MXFP8 reference require exact equality.
"""

import copy
import json
import os
from typing import Any

import pytest
import torch
import torch.distributed as dist
from torch.nn import functional as F

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.components.quantization.functional.npu_mxfp8 import MXFP8NpuOps
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.apply import apply_sharding_plan
from hyper_parallel.trainer.config import entries_to_plan_overrides
from hyper_parallel.trainer.config.resolver import resolve_config
from tests.common.grouped_experts_family import TinyMoE, family_config, replace_experts


class CountingMXOps(MXFP8NpuOps):
    """Count genuine float8 GMM operator calls without replacing arithmetic."""

    def __init__(self, backend: Any) -> None:
        """Validate the backend and start a fresh operator-call record."""
        super().__init__(backend)
        self.calls = []

    def quant_grouped_matmul(self, x1: torch.Tensor, x2: torch.Tensor, x2_scale: torch.Tensor,
                            **kwargs: Any) -> torch.Tensor:
        """Record float8 operands and execute the real device operator.

        Args:
            x1: Quantized left operand.
            x2: Quantized right operand.
            x2_scale: Right operand scales.
            **kwargs: Unmodified grouped operator options and left scales.
        """
        self.calls.append((str(x1.dtype), str(x2.dtype), kwargs["group_type"]))
        assert x1.dtype == torch.float8_e4m3fn and x2.dtype == torch.float8_e4m3fn
        return super().quant_grouped_matmul(x1, x2, x2_scale, **kwargs)


def _device():
    """Skip explicitly if real MXFP8 hardware/operators are unavailable."""
    backend = pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Real NPU unavailable; CPU contracts do not validate MXFP8 numerics")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    if not any(marker in torch.npu.get_device_name() for marker in ("Ascend950", "Ascend910_95")):
        pytest.skip("MXFP8 requires A5 hardware")
    return torch.device("npu", local_rank), backend


def _metric(name, actual, reference):
    """Report absolute and normalized error using a predeclared acceptance rule."""
    diff = (actual.float() - reference.float()).abs()
    rms = reference.float().square().mean().sqrt().item() if reference.numel() else 0.
    error = diff.square().mean().sqrt().item() if diff.numel() else 0.
    maximum = diff.max().item() if diff.numel() else 0.
    ref_max = reference.float().abs().max().item() if reference.numel() else 0.
    result = {"name": name, "max_abs": maximum, "nrmse": error / max(rms, 1e-12),
              "p99_abs": torch.quantile(diff.flatten(), .99).item() if diff.numel() else 0.}
    print(json.dumps(result), flush=True)
    assert torch.isfinite(actual).all()
    assert error <= .10 * rms + 1e-6, result
    assert maximum <= .5 * ref_max + 1e-6, result


def _bf16_local(experts, x, counts):
    """Plain reference with identical expert token groups and BF16 boundaries."""
    outputs = []
    for block, w1, w2 in zip(x.split(counts), experts.gate_up_proj, experts.down_proj):
        gate, up = (block @ w1).chunk(2, -1)
        outputs.append((F.silu(gate) * up) @ w2)
    return torch.cat(outputs)


def _check_local_case(expert, baseline, counts, frozen, device):
    """Compare one gradient-requirement/grouping combination with BF16."""
    for name, weight in expert.named_parameters():
        weight.requires_grad_(frozen not in (name, "all_weights"))
    reference = copy.deepcopy(baseline)
    for name, weight in reference.named_parameters():
        weight.requires_grad_(frozen not in (name, "all_weights"))
    x_low = torch.randn(sum(counts), 128, dtype=torch.bfloat16).to(device).requires_grad_(frozen != "input")
    x_ref = x_low.detach().clone().requires_grad_(x_low.requires_grad)
    groups = torch.tensor(counts, device=device, dtype=torch.int64)
    out = expert.forward_expert_major(x_low, groups)
    ref = _bf16_local(reference, x_ref, counts)
    upstream = torch.randn_like(out)
    named_variables = [(name, p) for name, p in [("input", x_low), *expert.named_parameters()]
                       if p.requires_grad]
    variables = [p for _, p in named_variables]
    ref_variables = [p for p in (x_ref, *reference.parameters()) if p.requires_grad]
    grads = torch.autograd.grad(out, variables, upstream, retain_graph=True)
    repeated = torch.autograd.grad(out, variables, upstream)
    expected = torch.autograd.grad(ref, ref_variables, upstream)
    label = f"counts={counts}/frozen={frozen}"
    _metric(label + "/output", out, ref)
    for (name, _), actual, again, wanted in zip(named_variables, grads, repeated, expected):
        torch.testing.assert_close(actual, again, rtol=0, atol=0)
        _metric(label + f"/{name}_grad", actual, wanted)
    if sum(counts) == 0:
        assert all(torch.count_nonzero(grad) == 0 for grad in grads)


def test_single_device():
    """Fixed routing, output/dX/dW, empty/frozen/repeated backward and three updates."""
    device, backend = _device()
    torch.manual_seed(793)
    source = TinyMoE(dtype=torch.bfloat16).to(device)
    baseline, low = copy.deepcopy(source), copy.deepcopy(source)
    replace_experts(baseline, family_config())
    replace_experts(low, family_config(use_mxfp8=True))
    ops = CountingMXOps(backend)
    low.mlp.experts.quantizer._npu_ops = ops
    tokens = 40
    indices = torch.tensor([[i % 4, (i + 1) % 4] for i in range(tokens)], device=device)
    scores = torch.tensor([[.25, .75]] * tokens, device=device, dtype=torch.bfloat16)
    x = torch.randn(1, tokens, 128, dtype=torch.bfloat16).to(device)
    for model in (source, baseline, low):
        model.mlp.gate.indices, model.mlp.gate.scores = indices, scores
    source_out, base_out, low_out = source(x), baseline(x), low(x)
    torch.testing.assert_close(base_out, source_out, rtol=1e-2, atol=1e-4)
    _metric("routed_output", low_out, base_out)
    assert len(ops.calls) == 2

    for counts in ([1, 0, 31, 33], [0, 0, 0, 0], [0, 32, 1, 31]):
        for frozen in ("none", "gate_up_proj", "down_proj", "all_weights", "input"):
            _check_local_case(low.mlp.experts, baseline.mlp.experts, counts, frozen, device)

    empty = torch.empty(1, 0, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    low.mlp.gate.indices = torch.empty(0, 2, device=device, dtype=torch.int64)
    low.mlp.gate.scores = torch.empty(0, 2, device=device, dtype=torch.bfloat16)
    low.zero_grad(set_to_none=True)
    low(empty).sum().backward()
    assert empty.grad.shape == empty.shape
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in low.parameters())
    low.mlp.gate.indices, low.mlp.gate.scores = indices, scores

    expert = low.mlp.experts
    expert.requires_grad_(True)
    optimizer = torch.optim.SGD(expert.parameters(), lr=.1)
    initial = [p.detach().clone() for p in expert.parameters()]
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        step_input = x.detach().clone().requires_grad_(True)
        output = low(step_input)
        loss = output.float().square().sum()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        assert step_input.grad is not None and torch.isfinite(step_input.grad).all()
    assert all(not torch.equal(before, after) for before, after in zip(initial, expert.parameters()))
    assert any(call[2] == 2 for call in ops.calls)
    torch.npu.synchronize()
    print(json.dumps({"single_device": "passed", "gmm_calls": len(ops.calls), "three_step_losses": losses}), flush=True)


def _ep_config():
    config = family_config(use_mxfp8=True).to_dict()
    config["plan_overrides"].append({
        "match": "mlp", "when": "ep", "region_dispatch": False,
        "local_compute_fn": {"_target_":
            "hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel.qwen3moe_ep_compute_fn"},
    })
    return resolve_config(config)


def _ordered_reference(experts, inputs, routes, scores):
    """Emulate dispatch ordering using tensor indexing, with no communication.

    Reproduce source-rank concatenation and the executor's (unstable) argsort
    on exactly the same local index arrays, then undo each permutation and
    reduce in the same per-sender dispatch order. MX block grouping is identical.
    """
    orders, sources = [], []
    for route in routes:
        flat_ids = route.flatten()
        orders.append(((flat_ids // 2) * 4 + flat_ids).argsort())
        sources.append(torch.arange(route.shape[0], device=route.device).repeat_interleave(2))
    returned = [[] for _ in inputs]
    for destination in range(2):
        pieces, ids, sizes = [], [], []
        for rank, route in enumerate(routes):
            order = orders[rank]
            selected = order[(route.flatten()[order] // 2) == destination]
            pieces.append(inputs[rank].reshape(-1, 128)[sources[rank][selected]])
            ids.append(route.flatten()[selected] - destination * 2)
            sizes.append(selected.numel())
        received, local_ids = torch.cat(pieces), torch.cat(ids)
        local_order = local_ids.argsort()
        counts = torch.bincount(local_ids, minlength=2)
        local = copy.deepcopy(experts)
        # Reference-only slicing; the tested model is sharded exclusively by the planner.
        local.gate_up_proj = torch.nn.Parameter(
            experts.gate_up_proj[destination * 2:(destination + 1) * 2].detach().clone())
        local.down_proj = torch.nn.Parameter(experts.down_proj[destination * 2:(destination + 1) * 2].detach().clone())
        sorted_output = local.forward_expert_major(received[local_order], counts)
        output = torch.empty_like(sorted_output)
        output[local_order] = sorted_output
        for rank, piece in enumerate(output.split(sizes)):
            returned[rank].append(piece)
        yield destination, local
    outputs = []
    for rank, parts in enumerate(returned):
        combined = torch.cat(parts)
        order = orders[rank]
        weighted = combined * scores[rank].flatten()[order, None].to(combined.dtype)
        result = torch.zeros_like(inputs[rank].reshape(-1, 128))
        result.index_add_(0, sources[rank][order], weighted)
        outputs.append(result.reshape_as(inputs[rank]))
    yield outputs


def _check_ep_case(model, reference_experts, rank, device, hooks, empty_rank):
    """Compare one routed distributed graph against the order-matched reference."""
    expert = model.mlp.experts
    torch.manual_seed(170 + int(empty_rank))
    inputs = [torch.randn(1, 35, 128, dtype=torch.bfloat16).to(device).requires_grad_() for _ in range(2)]
    routes = [torch.tensor([[i % 2, (i + 1) % 2] if empty_rank else
                            [(i + sender) % 4, (i + sender + 2) % 4]
                            for i in range(35)], device=device) for sender in range(2)]
    scores = [torch.tensor([[.137, .863]] * 35, device=device, dtype=torch.float32).requires_grad_()
              for _ in range(2)]
    upstream = [torch.randn_like(value) for value in inputs]
    reference = list(_ordered_reference(reference_experts, inputs, routes, scores))
    locals_by_rank, expected_outputs = [pair[1] for pair in reference[:-1]], reference[-1]
    reference_loss = sum((out * grad).sum() for out, grad in zip(expected_outputs, upstream))
    reference_loss.backward()
    actual_input = inputs[rank].detach().clone().requires_grad_()
    actual_scores = scores[rank].detach().clone().requires_grad_()
    model.mlp.gate.indices, model.mlp.gate.scores = routes[rank], actual_scores
    model.zero_grad(set_to_none=True)
    actual = model(actual_input)
    (actual * upstream[rank]).sum().backward()
    torch.testing.assert_close(actual, expected_outputs[rank], rtol=0, atol=0)
    torch.testing.assert_close(actual_input.grad, inputs[rank].grad, rtol=0, atol=0)
    torch.testing.assert_close(actual_scores.grad, scores[rank].grad, rtol=0, atol=0)
    for name, weight in expert.named_parameters():
        torch.testing.assert_close(weight.grad, getattr(locals_by_rank[rank], name).grad, rtol=0, atol=0)
    if empty_rank and rank == 1:
        assert hooks[-1] == (0, 0)
        assert all(torch.count_nonzero(p.grad) == 0 for p in expert.parameters())
    print(json.dumps({"rank": rank, "empty_receiving_rank": empty_rank,
                      "local_tokens": hooks[-1][0], "parity": "exact"}), flush=True)


def test_ep_two():
    """Real planner/apply + Qwen3 recipe; cross-rank routing and an empty receiving rank."""
    device, backend = _device()
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("Run with torchrun --nproc-per-node=2")
    dist.init_process_group("hccl")
    rank = dist.get_rank()
    try:
        torch.manual_seed(793)
        model = TinyMoE(dtype=torch.bfloat16).to(device)
        config = _ep_config()
        replace_experts(model, config, context={"ep": True, "ep_size": 2})
        reference_experts = copy.deepcopy(model.mlp.experts)
        mesh = init_device_mesh("npu", (2, 1), mesh_dim_names=("dp", "tp"))
        overrides = entries_to_plan_overrides(config.plan_overrides, ep_size=2)
        plan = ShardingPlanner(plan_overrides=overrides).plan(model, mesh, ep_size=2, sequence_parallel=False)
        print(f"rank={rank} plan={plan.explain()}", flush=True)
        model, shard_info = apply_sharding_plan(model, plan, mesh)
        expert = model.mlp.experts
        assert expert.gate_up_proj.shape == (2, 128, 512)
        assert expert.down_proj.shape == (2, 256, 128)
        assert expert.local_expert_count == 2 and expert.ep_use_grouped_gemm
        assert "mlp.experts.gate_up_proj" in shard_info
        ops = CountingMXOps(backend)
        expert.quantizer._npu_ops = ops
        hooks = []
        handle = expert.register_forward_hook(
            lambda module, args, output: hooks.append((args[0].shape[0], output.shape[0])))
        for empty_rank in (False, True):
            _check_ep_case(model, reference_experts, rank, device, hooks, empty_rank)
        handle.remove()
        assert len(hooks) == 2 and len(ops.calls) > 0
        print(json.dumps({"rank": rank, "gmm_calls": len(ops.calls), "ep_two": "passed"}), flush=True)
        torch.npu.synchronize()
    finally:
        dist.destroy_process_group()

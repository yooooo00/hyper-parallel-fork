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
"""GQA Family TP/CP numerical checks through replacement, planner and apply.

BF16 comparisons fix rtol=.03 and atol=.003 before execution. MXFP8 uses
existing local-operator limits: RMS error <= .10 * reference RMS + 1e-6,
maximum error <= .5 * reference maximum + 1e-6. Parameter SUMs below only
assemble the layer-level oracle; Trainer gradient reduction is a separate test.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any

import pytest
import torch
import torch.distributed as dist
from torch import nn
from transformers.models.llama.modeling_llama import eager_attention_forward

from hyper_parallel.components.quantization.quantizers.mxfp8 import MXFP8Quantizer
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.apply import apply_sharding_plan
from hyper_parallel.distributed.recipe_spec import TP
from hyper_parallel.models.replacement import _apply_module_replacement_actions
from hyper_parallel.trainer.config import entries_to_module_replacements, entries_to_plan_overrides
from hyper_parallel.trainer.config.resolver import resolve_config
from tests.common.gqa_family import attention_inputs, native_attention, replace_attention
from tests.torch.attention._test_mxfp8_gqa_attention import CountingLinearOps


_KERNEL_CALLS = []


def observed_attention(module: nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                       attention_mask: torch.Tensor | None, **kwargs: Any) -> tuple:
    """Observe the preserved kernel interface while executing its actual math.

    Args:
        module: Attention module whose configured kernel is observed.
        query: Query states passed unchanged to the selected kernel.
        key: Key states passed unchanged to the selected kernel.
        value: Value states passed unchanged to the selected kernel.
        attention_mask: Global causal or custom mask for the current query slice.
        kwargs: Keyword arguments forwarded unchanged to the observed operation.
    """
    _KERNEL_CALLS.append((query.shape[2], key.shape[2], value.shape[2]))
    return eager_attention_forward(module, query, key, value, attention_mask, **kwargs)


class _AttentionModel(nn.Module):
    """A native attention class with an ordinary planner-visible boundary."""

    def __init__(self, kind: str, bias: bool = False) -> None:
        """Construct native weights before the production replacement executor."""
        super().__init__()
        self.self_attn = native_attention(kind, bias=bias)
        self.config = self.self_attn.config

    def forward(self, hidden_states: torch.Tensor, position_embeddings: tuple,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Keep the outer call and the inner projection hooks intact.

        Args:
            hidden_states: Local hidden states entering the wrapped attention module.
            position_embeddings: Rotary cosine and sine tensors for the current positions.
            attention_mask: Global causal or custom mask for the current query slice.
        """
        return self.self_attn(hidden_states, position_embeddings=position_embeddings,
                              attention_mask=attention_mask)[0]


def _config(kind: str, low: bool, tp: int, cp: int, explicit: bool, bias: bool = False):
    """Resolve the actual public configuration; selectors never reach the baseline."""
    native_type = type(native_attention(kind))
    entries = [{
        "match": "self_attn", "module_type": f"{native_type.__module__}.{native_type.__qualname__}",
        "replace_module": {"_target_": "hyper_parallel.components.modules.GQAAttention",
                           "use_mxfp8": low, "use_fused_mxfp8": False,
                           "attention_interface": f"{__name__}.observed_attention"},
    }]
    if explicit:
        wrapper_path = "hyper_parallel.distributed.context_parallel.attention_interface.attention_interface_cp_wrapper"
        entries.append({"match": "self_attn", "inner_target": "self", "region_dispatch": False,
                        "inner_wrapper": {"_target_": wrapper_path}})
    return resolve_config({
        "model": {"_target_": f"{__name__}._AttentionModel", "kind": kind, "bias": bias},
        "optimizer": {"_target_": "torch.optim.SGD", "lr": .01},
        "accelerator": {"tp_size": tp, "cp_size": cp}, "plan_overrides": entries,
    })


def _check(name: str, actual: torch.Tensor, expected: torch.Tensor, low: bool) -> None:
    """Record errors while enforcing bounds fixed in this module's contract."""
    error = (actual.float() - expected.float()).abs()
    rms = expected.float().square().mean().sqrt().item()
    error_rms = error.square().mean().sqrt().item()
    record = {"name": name, "rank": dist.get_rank(), "mxfp8": low,
              "nrmse": error_rms / max(rms, 1e-12), "max_abs": error.max().item()}
    print(json.dumps(record), flush=True)
    assert torch.isfinite(actual).all(), record
    if low:
        assert error_rms <= .10 * rms + 1e-6, record
        assert error.max().item() <= .5 * expected.float().abs().max().item() + 1e-6, record
    else:
        torch.testing.assert_close(actual, expected, rtol=.03, atol=.003)


def _local_reference(value: torch.Tensor, placement, mesh, axis: str) -> torch.Tensor:
    """Index an expected tensor using the actual planner declaration."""
    if isinstance(placement, Shard):
        return value.chunk(mesh[axis].size(), dim=placement.dim)[mesh[axis].get_local_rank()]
    return value


def _check_weights(model, reference, layouts, mesh, tp: int) -> None:
    """Verify exact source values after the planner has performed TP partitioning."""
    for name, value in model.self_attn.named_parameters():
        expected = reference.get_parameter(name)
        if tp > 1:
            expected = _local_reference(expected, layouts[name][TP], mesh, "tp")
        actual = value.to_local() if isinstance(value, DTensor) else value
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _partitioned_reference(reference: nn.Module, inputs: dict, cp: int) -> tuple[torch.Tensor, list]:
    """Align projection batches, query chunks and gradient reductions with CP.

    Each replica calls the real GQA forward to prepare Q/K/V. The reference
    joins those tensors directly on one device; no distributed adapter or
    collective is involved, and the selected attention kernel stays intact.
    """
    replicas = [copy.deepcopy(reference) for _ in range(cp)]
    prepared = []
    kernel = reference.attention_interface

    def _capture(module, query, key, value, mask, **kwargs):
        """Intercept prepared states after the real projection, norm and RoPE."""
        del module
        prepared.append((query, key, value, mask, kwargs))
        return query.transpose(1, 2), None

    for index, replica in enumerate(replicas):
        replica.attention_interface = _capture
        replica(hidden_states=inputs["hidden_states"].chunk(cp, dim=1)[index],
                position_embeddings=tuple(value.chunk(cp, dim=1)[index] for value in inputs["position_embeddings"]),
                attention_mask=inputs["attention_mask"])
        replica.attention_interface = kernel
    keys = torch.cat([states[1] for states in prepared], dim=2)
    values = torch.cat([states[2] for states in prepared], dim=2)
    outputs = []
    for index, (replica, states) in enumerate(zip(replicas, prepared)):
        query, _, _, mask, kwargs = states
        mask = mask.chunk(cp, dim=-2)[index]
        output = kernel(replica, query, keys, values, mask, **kwargs)[0]
        outputs.append(replica.o_proj(output.flatten(-2)))
    return torch.cat(outputs, dim=1), replicas


def _parameter_gradients(model, reference, layouts, mesh, sizes: tuple[int, int], low: bool) -> None:
    """Assemble replicated partial gradients for the layer's unsharded oracle."""
    tp, cp = sizes
    for name, parameter in model.self_attn.named_parameters():
        assert parameter.grad is not None, name
        actual = parameter.grad.detach().clone()
        expected = torch.stack([replica.get_parameter(name).grad for replica in reference]).sum(dim=0)
        if cp > 1:
            dist.all_reduce(actual, group=mesh["cp"].get_group())
        if tp > 1:
            placement = layouts[name][TP]
            if isinstance(placement, Shard):
                expected = _local_reference(expected, placement, mesh, "tp")
            elif "norm" in name:
                dist.all_reduce(actual, group=mesh["tp"].get_group())
        _check(f"gradient/{name}", actual, expected, low)


def _inputs_for_rank(inputs: dict, mesh, cp: int) -> dict:
    """Slice contiguous sequence positions and rotary values; keep mask global."""
    if cp == 1:
        return inputs
    rank = mesh["cp"].get_local_rank()
    return {**inputs, "hidden_states": inputs["hidden_states"].chunk(cp, dim=1)[rank],
            "position_embeddings": tuple(value.chunk(cp, dim=1)[rank] for value in inputs["position_embeddings"])}


def _mask_inputs(inputs: dict, mode: str) -> torch.Tensor | None:
    """Keep an explicit additive oracle for default causal and boolean masks."""
    if mode == "none":
        return None
    if mode == "padding":
        padding = torch.ones(2, 64, dtype=torch.bool, device=inputs["hidden_states"].device)
        padding[0, -3:] = False
        padding[1, -5:] = False
        inputs["attention_mask"] = inputs["attention_mask"].masked_fill(~padding[:, None, None, :], float("-inf"))
        return padding
    if mode == "custom":
        allowed = inputs["attention_mask"] == 0
        allowed[..., 7::7] = False
        inputs["attention_mask"] = inputs["attention_mask"].masked_fill(~allowed, float("-inf"))
        return allowed
    return inputs["attention_mask"]


def _check_case(kind: str, low: bool, sizes: tuple[int, int], explicit: bool, mesh, device, backend,
                *, validate_mode: bool = False, mask_mode: str = "additive", bias: bool = False) -> None:
    """Build and compare one final Family selection on the actual dense mesh."""
    tp, cp = sizes
    torch.manual_seed(793)
    torch.npu.manual_seed(793)
    config = _config(kind, low, tp, cp, explicit, bias)
    model = config.model.build().to(device)
    reference, _ = replace_attention(copy.deepcopy(model.self_attn))
    _apply_module_replacement_actions(model, entries_to_module_replacements(config.plan_overrides), weights_mapping=[])
    overrides = entries_to_plan_overrides(config.plan_overrides, cp_size=cp)
    plan = ShardingPlanner(plan_overrides=overrides).plan(
        model, mesh, tp_size=config.accelerator.tp_size, cp_size=config.accelerator.cp_size, sequence_parallel=False
    )
    print(json.dumps({"kind": kind, "low": low, "rank": dist.get_rank(), "plan": plan.explain()}), flush=True)
    layouts = {
        f"{fqn}.{name}".removeprefix("self_attn."): placements
        for fqn, spec in plan.modules.items() for name, placements in (spec.params or {}).items()
    }
    model, _ = apply_sharding_plan(model, plan, mesh, validate_mode=validate_mode)
    divisor = 1 if validate_mode and cp == 1 else tp
    assert model.self_attn.num_heads == 4 // divisor
    assert model.self_attn.num_key_value_heads == 2 // divisor
    _check_weights(model, reference, layouts, mesh, tp)
    calls = []
    for projection in (model.self_attn.linear_qkv, model.self_attn.o_proj):
        projection.register_forward_hook(lambda module, args, output: calls.append(tuple(output.shape)))
    counter = CountingLinearOps(backend)
    if low:
        for projection in (model.self_attn.linear_qkv, model.self_attn.o_proj):
            projection.quantizer = MXFP8Quantizer(npu_ops=counter)
    inputs = attention_inputs(sequence=64, device=str(device))
    actual_mask = _mask_inputs(inputs, mask_mode)
    full_hidden = inputs["hidden_states"].detach().clone().requires_grad_()
    expected, gradient_references = _partitioned_reference(reference, {**inputs, "hidden_states": full_hidden}, cp)
    upstream = torch.randn_like(expected)
    (expected * upstream).sum().backward()
    local_inputs = _inputs_for_rank(inputs, mesh, cp)
    local_hidden = local_inputs["hidden_states"].detach().clone().requires_grad_()
    _KERNEL_CALLS.clear()
    actual = model(**{**local_inputs, "hidden_states": local_hidden, "attention_mask": actual_mask})
    assert _KERNEL_CALLS == [(64 // cp, 64, 64)]
    cp_rank = mesh["cp"].get_local_rank() if cp > 1 else 0
    local_upstream = upstream.chunk(cp, dim=1)[cp_rank]
    (actual * local_upstream).sum().backward()
    _check("output", actual, expected.chunk(cp, dim=1)[cp_rank], low)
    _check("input_gradient", local_hidden.grad, full_hidden.grad.chunk(cp, dim=1)[cp_rank], low)
    _parameter_gradients(model, gradient_references, layouts, mesh, sizes, low)
    assert len(calls) == 2
    assert counter.calls == (6 if low else 0)
    print(json.dumps({"kind": kind, "sizes": sizes, "explicit": explicit, "mxfp8": low,
                      "rank": dist.get_rank(), "fp8_calls": counter.calls, "projection_hooks": calls}), flush=True)


@pytest.mark.parametrize("kind", ["llama", "qwen3_moe"])
@pytest.mark.parametrize("topology", ["tp", "cp", "tp_cp"])
@pytest.mark.parametrize("explicit", [False, True], ids=["default", "explicit"])
def test_gqa_parallel(kind: str, topology: str, explicit: bool) -> None:
    """Compare both BF16 and MXFP8 through the unchanged TP/CP planner path.

    Args:
        kind: Native attention protocol: Llama or Qwen3-MoE.
        topology: Dense parallel topology selected by the real planner.
        explicit: Whether the recipe explicitly supplies the CP adapter.
    """
    backend = pytest.importorskip("torch_npu")
    if explicit and topology == "tp":
        pytest.skip("Explicit CP adapter requires CP; TP is covered by the default case")
    sizes = {"tp": (2, 1), "cp": (1, 2), "tp_cp": (2, 2)}[topology]
    assert int(os.environ["WORLD_SIZE"]) == sizes[0] * sizes[1]
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        mesh = init_device_mesh("npu", (sizes[1], sizes[0]), mesh_dim_names=("cp", "tp"))
        for low in (False, True):
            _check_case(kind, low, sizes, explicit, mesh, device, backend)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("kind", ["llama", "qwen3_moe"])
def test_gqa_tp_validate(kind: str) -> None:
    """Reject the observed mixed RoPE DTensor gap during the actual build.

    Args:
        kind: Native attention protocol: Llama or Qwen3-MoE.
    """
    backend = pytest.importorskip("torch_npu")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        mesh = init_device_mesh("npu", (2,), mesh_dim_names=("tp",))
        with pytest.raises(NotImplementedError, match="self_attn.*RoPE DTensor dispatch"):
            _check_case(kind, False, (2, 1), False, mesh, device, backend, validate_mode=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("kind", ["llama", "qwen3_moe"])
def test_gqa_cp_masks(kind: str) -> None:
    """Verify causal defaults and boolean padding/custom masks on both protocols.

    Args:
        kind: Native attention protocol: Llama or Qwen3-MoE.
    """
    backend = pytest.importorskip("torch_npu")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        mesh = init_device_mesh("npu", (2,), mesh_dim_names=("cp",))
        for mode in ("none", "padding", "custom"):
            _check_case(kind, False, (1, 2), False, mesh, device, backend, mask_mode=mode)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_gqa_tp_bias() -> None:
    """Both projection classes preserve one row-projection bias after TP SUM."""
    backend = pytest.importorskip("torch_npu")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        mesh = init_device_mesh("npu", (2,), mesh_dim_names=("tp",))
        for low in (False, True):
            _check_case("llama", low, (2, 1), False, mesh, device, backend, bias=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()

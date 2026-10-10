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
"""MLA Family CP with distinct QK/value widths and real NPU RoPE gradients."""

from __future__ import annotations

import copy
import os
from typing import Any

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.apply import apply_sharding_plan
from hyper_parallel.models.replacement import _apply_module_replacement_actions
from hyper_parallel.trainer.config import entries_to_module_replacements, entries_to_plan_overrides
from hyper_parallel.trainer.config.resolver import resolve_config
from tests.torch.context_parallel._test_attention_family import _assert_close


_INTERFACE_CALLS = []


def selected_attention(module: nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                       attention_mask: torch.Tensor | None, *, dropout: float = 0., scaling: float | None = None,
                       **kwargs: Any) -> tuple[torch.Tensor, None]:
    """Preserve the selected scaled dot-product attention interface with different QK/V widths.

    Args:
        module: Attention module whose configured kernel is observed.
        query: Query states passed unchanged to the selected kernel.
        key: Key states passed unchanged to the selected kernel.
        value: Value states passed unchanged to the selected kernel.
        attention_mask: Global causal or custom mask for the current query slice.
        dropout: Attention dropout probability passed to the original kernel.
        scaling: Explicit attention-logit scale, if provided.
        kwargs: Keyword arguments forwarded unchanged to the observed operation.
    """
    del kwargs
    _INTERFACE_CALLS.append((tuple(query.shape), tuple(key.shape), tuple(value.shape)))
    output = F.scaled_dot_product_attention(  # pylint: disable=not-callable
        query, key, value, attn_mask=attention_mask, dropout_p=dropout,
        is_causal=attention_mask is None and module.is_causal, scale=scaling,
    )
    return output.transpose(1, 2), None


class _MLAModel(nn.Module):
    """Small actual DeepSeek attention source for the production replacement."""

    def __init__(self) -> None:
        """Build an actual DeepSeekV3 attention source on the CPU."""
        super().__init__()
        self.config = DeepseekV3Config.from_dict({
            "hidden_size": 64, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 2,
            "q_lora_rank": 16, "kv_lora_rank": 16,
            "qk_nope_head_dim": 16, "qk_rope_head_dim": 16, "v_head_dim": 16,
            "attention_dropout": 0.,
        })
        self.self_attn = DeepseekV3Attention(self.config, layer_idx=0)

    def forward(self, hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Use the native layer argument and result contract.

        Args:
            hidden_states: Local hidden states entering the wrapped attention module.
            position_embeddings: Rotary cosine and sine tensors for the current positions.
        """
        return self.self_attn(hidden_states, position_embeddings=position_embeddings)[0]


def _config(explicit):
    """Parse replacement and explicit/default CP configuration together."""
    entry = {
        "match": "self_attn",
        "module_type": "transformers.models.deepseek_v3.modeling_deepseek_v3.DeepseekV3Attention",
        "region_dispatch": False,
        "replace_module": {
            "_target_": "hyper_parallel.components.modules.MLAAttention",
            "attention_interface": "tests.torch.context_parallel._test_mla_family.selected_attention",
        },
    }
    if explicit:
        entry["inner_wrapper"] = {
            "_target_": "hyper_parallel.distributed.attention_family.mla.mla_attention_cp_wrapper"
        }
    return resolve_config({
        "model": {"_target_": "torch.nn.Identity"},
        "optimizer": {"_target_": "torch.optim.SGD", "lr": .01},
        "plan_overrides": [entry],
    })


@pytest.mark.parametrize("explicit", [True, False], ids=["explicit", "default"])
def test_mla_family_cp(explicit):
    """Retain loaded latents and compare output, dX and every projection gradient.

    Args:
        explicit: Whether the recipe explicitly supplies the CP adapter.
    """
    pytest.importorskip("torch_npu")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        torch.manual_seed(793)
        model = _MLAModel().to(device).train()
        source_weight = torch.cat((model.self_attn.q_a_proj.weight, model.self_attn.kv_a_proj_with_mqa.weight))
        config = _config(explicit)
        _apply_module_replacement_actions(
            model, entries_to_module_replacements(config.plan_overrides), weights_mapping=[],
        )
        torch.testing.assert_close(model.self_attn.linear_qkv.weight, source_weight, atol=0, rtol=0)
        reference = copy.deepcopy(model)
        inputs = torch.randn(2, 32, 64, device=device)
        angles = torch.randn(2, 32, 8, device=device)
        angles = torch.cat((angles, angles), dim=-1)
        position_embeddings = (angles.cos(), angles.sin())
        upstream = torch.randn_like(inputs)
        full_inputs = inputs.detach().clone().requires_grad_()
        expected = reference(full_inputs, position_embeddings)
        (expected * upstream).sum().backward()
        mesh = init_device_mesh("npu", (2,), mesh_dim_names=("cp",))
        plan = ShardingPlanner(plan_overrides=entries_to_plan_overrides(config.plan_overrides)).plan(
            model, mesh, cp_size=2, sequence_parallel=False,
        )
        assert plan.modules["self_attn"].inner_wrapper is not None
        model, _ = apply_sharding_plan(model, plan, mesh)
        assert model.self_attn._hp_attention_cp_original_interface is selected_attention
        rank = dist.get_rank()
        local_inputs = inputs.chunk(2, dim=1)[rank].detach().clone().requires_grad_()
        local_rope = tuple(tensor.chunk(2, dim=1)[rank] for tensor in position_embeddings)
        actual = model(local_inputs, local_rope)
        (actual * upstream.chunk(2, dim=1)[rank]).sum().backward()
        _assert_close("mla_output", actual, expected.chunk(2, dim=1)[rank])
        _assert_close("mla_input_gradient", local_inputs.grad, full_inputs.grad.chunk(2, dim=1)[rank])
        expected_params = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            gradient = parameter.grad.clone()
            dist.all_reduce(gradient)
            _assert_close(name, gradient, expected_params[name].grad)
        assert _INTERFACE_CALLS[-1][0][2] == 16
        assert _INTERFACE_CALLS[-1][1][2] == 32
        assert _INTERFACE_CALLS[-1][1][-1] != _INTERFACE_CALLS[-1][2][-1]
        dist.barrier()
    finally:
        dist.destroy_process_group()

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
"""Omni-backed DSA CP acceptance; missing operators are explicit skips.

The BF16 communication tolerance is fixed before execution at rtol=.02,
atol=.002. Index selection must exactly match the global causal reference.
Indexer KL auxiliary training is a separate, currently unvalidated contract.
"""

import copy
import importlib
import os
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch import nn
from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Attention

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.apply import apply_sharding_plan
from hyper_parallel.models.replacement import _apply_module_replacement_actions
from hyper_parallel.trainer.config import entries_to_module_replacements, entries_to_plan_overrides
from hyper_parallel.trainer.config.resolver import resolve_config


class _DSAModel(nn.Module):
    """Actual DeepSeekV3.2 source retaining default sparse-kernel dimensions."""

    def __init__(self) -> None:
        """Reduce projection input dimensions while preserving kernel head geometry."""
        super().__init__()
        self.config = DeepseekV32Config.from_dict({
            "hidden_size": 128, "q_lora_rank": 128, "num_hidden_layers": 1,
            "dsa_loss_coeff": 0., "attention_dropout": 0.,
        })
        self.self_attn = DeepseekV32Attention(self.config, layer_idx=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Keep the original attention call and return contract.

        Args:
            hidden_states: Local hidden states entering the wrapped attention module.
        """
        return self.self_attn(hidden_states)[0]


def _configuration(explicit):
    entry = {
        "match": "self_attn",
        "module_type": "transformers.models.deepseek_v32.modeling_deepseek_v32.DeepseekV32Attention",
        "region_dispatch": False,
        "replace_module": {"_target_": "hyper_parallel.components.modules.DeepseekV32DSAAttention"},
    }
    if explicit:
        entry["inner_wrapper"] = {
            "_target_": "hyper_parallel.distributed.attention_family.dsa.dsa_attention_cp_wrapper"
        }
    return resolve_config({
        "model": {"_target_": "torch.nn.Identity"},
        "optimizer": {"_target_": "torch.optim.SGD", "lr": .01},
        "plan_overrides": [entry],
    })


@pytest.mark.parametrize("explicit", [True, False], ids=["explicit", "default"])
def test_dsa_family_cp(explicit):
    """Compare sparse selection and main attention gradients with global causality.

    Args:
        explicit: Whether the recipe explicitly supplies the CP adapter.
    """
    pytest.importorskip("omni_training_custom_ops", reason="DSA requires actual Omni sparse kernels")
    pytest.importorskip("torch_npu")
    modules = importlib.import_module("hyper_parallel.components.modules.dsa_attention")
    adapter = importlib.import_module("hyper_parallel.distributed.attention_family.dsa")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        torch.manual_seed(793)
        model = _DSAModel().to(device=device, dtype=torch.bfloat16).train()
        config = _configuration(explicit)
        source_weight = torch.cat((model.self_attn.q_a_proj.weight, model.self_attn.kv_a_proj_with_mqa.weight))
        _apply_module_replacement_actions(
            model, entries_to_module_replacements(config.plan_overrides), weights_mapping=[],
        )
        torch.testing.assert_close(model.self_attn.linear_qkv.weight, source_weight, atol=0, rtol=0)
        reference = copy.deepcopy(model)
        inputs = torch.randn(1, 256, 128, device=device, dtype=torch.bfloat16)
        upstream = torch.randn_like(inputs)
        captured = []
        original_indexer = modules.dsa_indexer

        def observed_indexer(*args: Any, **kwargs: Any) -> tuple:
            """Observe actual device indices without changing sparse arithmetic.

            Args:
                args: Positional arguments forwarded unchanged to the observed operation.
                kwargs: Keyword arguments forwarded unchanged to the observed operation.
            """
            result = original_indexer(*args, **kwargs)
            captured.append(result[0].detach().clone())
            return result

        full_inputs = inputs.detach().clone().requires_grad_()
        with patch.object(modules, "dsa_indexer", observed_indexer):
            expected = reference(full_inputs)
        (expected * upstream).sum().backward()
        mesh = init_device_mesh("npu", (2,), mesh_dim_names=("cp",))
        plan = ShardingPlanner(plan_overrides=entries_to_plan_overrides(config.plan_overrides)).plan(
            model, mesh, cp_size=2, sequence_parallel=False,
        )
        model, _ = apply_sharding_plan(model, plan, mesh)
        assert model.self_attn._hp_deepseek_v32_dsa_cp_config["cp_size"] == 2
        rank = dist.get_rank()
        local_inputs = inputs.chunk(2, dim=1)[rank].detach().clone().requires_grad_()
        with patch.object(adapter, "dsa_indexer", observed_indexer):
            actual = model(local_inputs)
        (actual * upstream.chunk(2, dim=1)[rank]).sum().backward()
        torch.testing.assert_close(captured[1], captured[0].chunk(2, dim=0)[rank], atol=0, rtol=0)
        torch.testing.assert_close(actual, expected.chunk(2, dim=1)[rank], rtol=.02, atol=.002)
        torch.testing.assert_close(local_inputs.grad, full_inputs.grad.chunk(2, dim=1)[rank], rtol=.02, atol=.002)
        expected_params = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            if "indexer." in name:
                assert parameter.grad is None and expected_params[name].grad is None
                continue
            assert parameter.grad is not None, name
            gradient = parameter.grad.clone()
            dist.all_reduce(gradient)
            torch.testing.assert_close(gradient, expected_params[name].grad, rtol=.02, atol=.002)
        dist.barrier()
    finally:
        dist.destroy_process_group()

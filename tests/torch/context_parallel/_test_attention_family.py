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
"""Real-device Family adapters against unsharded native training modules.

FP32 communication comparisons fix rtol=3e-4 and atol=3e-5 before execution.
These tolerate normal reduction ordering; they are not low-precision criteria.
"""

import copy
import faulthandler
import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

from hyper_parallel.components.modules.kimi_delta_attention import KimiDeltaAttention
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.apply import apply_sharding_plan
from hyper_parallel.trainer.config import entries_to_plan_overrides
from hyper_parallel.trainer.config.resolver import resolve_config


class _AttentionModel(nn.Module):
    """Minimal native attention model retaining actual Family class identity."""

    def __init__(self, family: str) -> None:
        """Create one native state-attention module with small dimensions."""
        super().__init__()
        if family == "gdn":
            config = Qwen3_5TextConfig.from_dict({
                "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
                "num_attention_heads": 2, "num_key_value_heads": 2,
                "linear_num_key_heads": 2, "linear_num_value_heads": 2,
                "linear_key_head_dim": 8, "linear_value_head_dim": 8,
                "linear_conv_kernel_dim": 4,
            })
            self.self_attn = Qwen3_5GatedDeltaNet(config, layer_idx=0)
            self.config = config
        else:
            self.self_attn = KimiDeltaAttention(
                hidden_size=32, num_heads=2, head_k_dim=8, head_v_dim=8,
                conv_kernel_size=4, chunk_size=64,
            )
            self.config = SimpleNamespace(
                model_type="kimi_k3", architectures=["KimiK3ForCausalLM"],
                num_attention_heads=2, num_key_value_heads=2, hidden_size=32,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Execute the original module call, preserving child hooks.

        Args:
            hidden_states: Local hidden states entering the wrapped attention module.
        """
        return self.self_attn(hidden_states)


def _config(family: str, explicit: bool):
    """Parse actual public plan entries, leaving Family default resolution intact."""
    entry = {"match": "self_attn", "region_dispatch": False}
    if explicit:
        wrapper = "gated_delta_net" if family == "gdn" else "kimi_delta_attention"
        entry["inner_wrapper"] = {
            "_target_": f"hyper_parallel.distributed.attention_family.state.{wrapper}_cp_wrapper"
        }
    return resolve_config({
        "model": {"_target_": "torch.nn.Identity"},
        "optimizer": {"_target_": "torch.optim.SGD", "lr": .01},
        "plan_overrides": [entry],
    })


def _assert_close(name, actual, reference):
    """Report and enforce the predeclared FP32 communication tolerance."""
    error = (actual - reference).float()
    metric = {"name": name, "rank": dist.get_rank(),
              "max_abs": error.abs().max().item(),
              "rms": error.square().mean().sqrt().item()}
    print(json.dumps(metric), flush=True)
    torch.testing.assert_close(actual, reference, rtol=3e-4, atol=3e-5)


@pytest.mark.parametrize("family", ["gdn", "kda"])
@pytest.mark.parametrize("explicit", [True, False], ids=["explicit", "default"])
def test_state_family_cp(family, explicit):
    """Compare CP2 outputs, cross-boundary input gradients and every parameter.

    Args:
        family: Stateful attention Family selected by the replacement config.
        explicit: Whether the recipe explicitly supplies the CP adapter.
    """
    faulthandler.dump_traceback_later(60)
    pytest.importorskip("torch_npu")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        torch.manual_seed(793)
        model = _AttentionModel(family).to(device).train()
        reference = copy.deepcopy(model)
        inputs = torch.randn(2, 128, 32, device=device)
        upstream = torch.randn_like(inputs)
        full_inputs = inputs.detach().clone().requires_grad_()
        reference_output = reference(full_inputs)
        (reference_output * upstream).sum().backward()
        mesh = init_device_mesh("npu", (2,), mesh_dim_names=("cp",))
        config = _config(family, explicit)
        plan = ShardingPlanner(plan_overrides=entries_to_plan_overrides(config.plan_overrides)).plan(
            model, mesh, cp_size=2, sequence_parallel=False,
        )
        assert plan.modules["self_attn"].inner_wrapper is not None
        model, _ = apply_sharding_plan(model, plan, mesh)
        marker = getattr(model.self_attn, f"_hp_{family}_cp_config")
        assert marker["mode"] == "p2p" and marker["backend"] == "eager"
        rank = dist.get_rank()
        local_inputs = inputs.chunk(2, dim=1)[rank].detach().clone().requires_grad_()
        actual = model(local_inputs)
        (actual * upstream.chunk(2, dim=1)[rank]).sum().backward()
        _assert_close("output", actual, reference_output.chunk(2, dim=1)[rank])
        _assert_close("input_gradient", local_inputs.grad, full_inputs.grad.chunk(2, dim=1)[rank])
        expected_params = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            gradient = parameter.grad.clone()
            # Planner local tensor mode leaves replicated parameter reduction to
            # the training runtime. Sum the partial gradients for this layer oracle.
            dist.all_reduce(gradient)
            _assert_close(name, gradient, expected_params[name].grad)
        dist.barrier()
    finally:
        dist.destroy_process_group()
        faulthandler.cancel_dump_traceback_later()

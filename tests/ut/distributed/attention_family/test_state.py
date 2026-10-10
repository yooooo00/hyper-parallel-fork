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
"""CPU contracts for actual state Family defaults and shared atomic adapters."""

import unittest
from typing import Any
from unittest.mock import patch

import torch
from torch import nn

from hyper_parallel.components.modules.family import get_module_family_selection
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.attention_family import state
from hyper_parallel.trainer.config import entries_to_plan_overrides
from hyper_parallel.trainer.config.resolver import resolve_config
from tests.torch.context_parallel._test_attention_family import _AttentionModel, _config
from tests.ut.auto_models.distributed.conftest import FakeDeviceMesh


class _Executor(nn.Module):
    """Device isolation only; selection and rewrite logic remain production code."""

    def __init__(self, module: nn.Module, mesh: Any, **kwargs: Any) -> None:
        """Record the layer without initializing a distributed executor."""
        super().__init__()
        self.module = module
        self.mesh = mesh
        self.options = kwargs

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Avoid communication in the hardware-independent rewrite test."""
        del kwargs
        return hidden_states


class StateFamilyTests(unittest.TestCase):
    """Validate exact class registration and normal planner default resolution."""

    def test_default_and_explicit_real_classes(self):
        """Both state protocols resolve their adapters through the real planner."""
        for family in ("gdn", "kda"):
            for explicit in (False, True):
                with self.subTest(family=family, explicit=explicit):
                    model = _AttentionModel(family)
                    selection = get_module_family_selection(model.self_attn)
                    self.assertIsNotNone(selection)
                    config = _config(family, explicit)
                    plan = ShardingPlanner(plan_overrides=entries_to_plan_overrides(config.plan_overrides)).plan(
                        model, FakeDeviceMesh((2,), ("cp",)), cp_size=2,
                    )
                    spec = plan.modules["self_attn"]
                    self.assertIsNotNone(spec.inner_wrapper)
                    self.assertFalse(spec.region_dispatch)
                    self.assertIn("self_attn", plan.family_modules)

    def test_explicit_disable_is_not_reenabled(self):
        """An explicitly disabled required adapter fails with the module path."""
        for family in ("gdn", "kda"):
            with self.subTest(family=family):
                config = resolve_config({
                    "model": {"_target_": "torch.nn.Identity"},
                    "optimizer": {"_target_": "torch.optim.SGD", "lr": .01},
                    "plan_overrides": [{"match": "self_attn", "inner_wrapper": "none"}],
                })
                with self.assertRaisesRegex(ValueError, "self_attn.*explicitly disabled"):
                    ShardingPlanner(plan_overrides=entries_to_plan_overrides(config.plan_overrides)).plan(
                        _AttentionModel(family), FakeDeviceMesh((2,), ("cp",)), cp_size=2,
                    )

    def test_state_rewrite_is_atomic_and_rejects_cache_and_packed(self):
        """The state boundary keeps parameters and checks cache aliases before execution."""
        module = _AttentionModel("kda").self_attn
        original = module.forward
        with patch.object(state, "KimiDeltaAttentionLayerP2PCP", _Executor):
            request = state.kimi_delta_attention_cp_wrapper(
                module, None, None, FakeDeviceMesh((2,), ("cp",)), None,
            )
        self.assertEqual(module.forward, original)
        self.assertNotIn("_hp_kda_cp_config", vars(module))
        self.assertEqual(request.companion_attrs["_hp_kda_cp_config"]["mode"], "p2p")
        inputs = torch.randn(1, 4, 32)
        for kwargs in ({"cache_params": object()}, {"use_cache": True}, {"actual_seq_len": [4]}):
            with self.subTest(kwargs=tuple(kwargs)), self.assertRaisesRegex(ValueError, "cache|packed"):
                request.forward(inputs, **kwargs)
        self.assertIs(request.forward(inputs), inputs)

    def test_all_packed_aliases_fail_before_state_execution(self) -> None:
        """Neither default state adapter may join separately packed sequences."""
        aliases = (
            "packed_seq_params", "actual_seq_len", "actual_q_len", "actual_kv_len",
            "actual_seq_qlen", "actual_seq_kvlen", "cu_seq_lens", "cu_seq_lens_q",
            "cu_seq_lens_k", "cu_seq_lens_kv", "cu_seqlens", "cu_seqlens_q",
            "cu_seqlens_k", "cu_seqlens_kv", "seq_idx",
        )
        adapters = (
            ("gdn", "GatedDeltaNetP2PCP", state.gated_delta_net_cp_wrapper),
            ("kda", "KimiDeltaAttentionLayerP2PCP", state.kimi_delta_attention_cp_wrapper),
        )
        for family, executor_name, wrapper in adapters:
            module = _AttentionModel(family).self_attn
            with patch.object(state, executor_name, _Executor):
                request = wrapper(module, None, None, FakeDeviceMesh((2,), ("cp",)), None)
            with patch.object(_Executor, "forward") as execute:
                for alias in aliases:
                    with self.subTest(family=family, alias=alias):
                        with self.assertRaisesRegex(ValueError, f"{family.upper()}.*packed.*{alias}"):
                            request.forward(torch.empty(1, 4, 32), **{alias: [0, 2, 4]})
                execute.assert_not_called()

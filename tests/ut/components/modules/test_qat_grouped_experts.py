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
"""QAT Family selects the existing strategies through real configuration."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from hyper_parallel.components.modules.low_precision_grouped_experts import LowPrecisionGroupedExperts
from hyper_parallel.components.quantization.functional.mxfp8_gmm_func import MXFP8GroupedLinear
from hyper_parallel.components.quantization.functional.w4a8_gmm_func import W4A8GroupedLinear
from hyper_parallel.components.quantization.functional.fake_w4a8_gmm_func import FakeW4A8GroupedLinear
from hyper_parallel.trainer.config.resolver import replace_override_path, resolve_config
from hyper_parallel.trainer.config.parallelism import entries_to_module_replacements
from tests.common.grouped_experts_family import TinyMoE
from hyper_parallel.components.quantization.modules.mxfp8_linear import MXFP8Linear
from tests.common.mark_utils import arg_mark
from tests.common.linear_family import linear_config
from tests.common.qat_family import install_family, qat_config


class TestQATGroupedExperts(unittest.TestCase):
    """Mock device capability checks only; registration and strategy selection stay real."""

    def setUp(self) -> None:
        """Isolate NPU availability without replacing the selected computation strategy."""
        for name in ("validate_npu_gmm_runtime", "validate_w4a8_gmm_runtime", "validate_fake_w4a8_gmm_runtime"):
            probe = patch("hyper_parallel.components.modules.low_precision_grouped_experts." + name)
            probe.start()
            self.addCleanup(probe.stop)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_named_scheme_selects_strategy_and_preserves_weights(self) -> None:
        """
        Feature: Scheme-selected Module Family.
        Description: Named schemes choose their native/fake strategy and retain canonical parameters.
        Expectation: Selection, validation and parameter behavior match the stated contract.
        """
        cases = [(False, 32, "mxfp8", MXFP8GroupedLinear)]
        cases += [(fake, block, "mxfp4", FakeW4A8GroupedLinear if fake else W4A8GroupedLinear)
                  for fake in (False, True) for block in (32, 128)]
        for fake, block, fmt, expected in cases:
            with self.subTest(fake=fake, block=block, fmt=fmt):
                model = TinyMoE(canonical=True)
                weight = model.mlp.experts.gate_up_proj
                weight.requires_grad_(False)
                install_family(model, qat_config(fake=fake, block_size=block, weight_format=fmt))
                self.assertIsInstance(model.mlp.experts, LowPrecisionGroupedExperts)
                self.assertIs(type(model.mlp.experts.grouped_linear), expected)
                self.assertIs(model.mlp.experts.gate_up_proj, weight)
                self.assertFalse(weight.requires_grad)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_scheme_conflicts_and_unknown_arguments(self) -> None:
        """
        Feature: Scheme-selected Module Family.
        Description: Invalid choices fail in plan construction before any device computation.
        Expectation: Selection, validation and parameter behavior match the stated contract.
        """
        for args in ({"use_mxfp8": True}, {"use_fused_mxfp8": True}, {"unknown_option": 1}):
            with self.subTest(args=args):
                raw = qat_config().to_dict()
                raw["plan_overrides"][0]["replace_module"].update(args)
                config = resolve_config(raw)
                with self.assertRaises((ValueError, TypeError)):
                    entries_to_module_replacements(config.plan_overrides,
                                                   low_precision_config=config.training.low_precision)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_scheme_name_roundtrip_and_override(self) -> None:
        """
        Feature: Scheme-selected Module Family.
        Description: Names serialize unchanged; changing the reference selects by its contents.
        Expectation: Selection, validation and parameter behavior match the stated contract.
        """
        raw = qat_config().to_dict()
        raw["training"]["low_precision"]["dtype_schemes"]["different_name"] = {
            "weight_format": "mxfp8", "act_format": "mxfp8"}
        config = resolve_config(raw)
        config.plan_overrides[0].replace_module = replace_override_path(
            config.plan_overrides[0].replace_module, ["low_precision_dtype_scheme"],
            "different_name", path="replace_module")
        config = resolve_config(config.to_dict())
        self.assertFalse(config.training.low_precision.enabled)
        self.assertIsNone(config.plan_overrides[0].when)
        self.assertEqual(config.plan_overrides[0].replace_module.low_precision_dtype_scheme, "different_name")
        model = install_family(TinyMoE(), config)
        self.assertIs(type(model.mlp.experts.grouped_linear), MXFP8GroupedLinear)

    @patch("hyper_parallel.components.quantization.modules.mxfp8_linear.validate_npu_runtime")
    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_mixed_dense_and_expert_schemes(self, probe) -> None:
        """
        Feature: Scheme-selected Module Family.
        Description: One configuration independently binds a Dense and a grouped scheme.
        Expectation: Selection, validation and parameter behavior match the stated contract.
        """
        raw = qat_config(fake=True).to_dict()
        dense = linear_config(quantized=True).to_dict()
        raw["training"]["low_precision"]["dtype_schemes"].update(
            dense["training"]["low_precision"]["dtype_schemes"])
        entry = dense["plan_overrides"][0]
        entry["match"] = "head"
        raw["plan_overrides"].append(entry)
        model = TinyMoE()
        model.head = nn.Linear(128, 256)
        install_family(model, resolve_config(raw))
        self.assertIs(type(model.head), MXFP8Linear)
        self.assertIs(type(model.mlp.experts.grouped_linear), FakeW4A8GroupedLinear)
        probe.assert_called_once()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_transposed_layout_preserves_loaded_values(self) -> None:
        """
        Feature: Scheme-selected Module Family.
        Description: Noncanonical source weights are copied by the existing high-performance parent.
        Expectation: Selection, validation and parameter behavior match the stated contract.
        """
        model = TinyMoE()
        before = model.mlp.experts.gate_up_proj.detach().clone()
        install_family(model, qat_config())
        torch.testing.assert_close(model.mlp.experts.gate_up_proj, before.transpose(-2, -1))

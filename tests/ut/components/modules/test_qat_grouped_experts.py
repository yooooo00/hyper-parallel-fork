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

from hyper_parallel.components.modules.qat_grouped_experts import QATGroupedExperts
from hyper_parallel.components.quantization.functional.mxfp8_gmm_func import MXFP8GroupedLinear
from hyper_parallel.components.quantization.functional.w4a8_gmm_func import W4A8GroupedLinear
from hyper_parallel.components.quantization.functional.fake_w4a8_gmm_func import FakeW4A8GroupedLinear
from hyper_parallel.trainer.config.resolver import ConfigResolutionError
from tests.common.grouped_experts_family import TinyMoE, family_config
from tests.common.qat_family import install_qat, qat_config


class TestQATGroupedExperts(unittest.TestCase):
    """Mock device capability checks only; registration and strategy selection stay real."""

    def setUp(self) -> None:
        """Isolate NPU availability without replacing the selected computation strategy."""
        for name in ("validate_npu_gmm_runtime", "validate_w4a8_gmm_runtime", "validate_fake_w4a8_gmm_runtime"):
            probe = patch("hyper_parallel.components.quantization.functional.strategy_factory." + name)
            probe.start()
            self.addCleanup(probe.stop)

    def test_named_scheme_selects_strategy_and_preserves_weights(self) -> None:
        """Named schemes choose their native/fake strategy and retain canonical parameters."""
        cases = [(False, 32, "mxfp8", MXFP8GroupedLinear)]
        cases += [(fake, block, "mxfp4", FakeW4A8GroupedLinear if fake else W4A8GroupedLinear)
                  for fake in (False, True) for block in (32, 128)]
        for fake, block, fmt, expected in cases:
            with self.subTest(fake=fake, block=block, fmt=fmt):
                model = TinyMoE(canonical=True)
                weight = model.mlp.experts.gate_up_proj
                weight.requires_grad_(False)
                install_qat(model, qat_config(fake=fake, block_size=block, weight_format=fmt))
                self.assertIsInstance(model.mlp.experts, QATGroupedExperts)
                self.assertIs(type(model.mlp.experts.grouped_linear), expected)
                self.assertIs(model.mlp.experts.gate_up_proj, weight)
                self.assertFalse(weight.requires_grad)

    def test_conflicts_are_not_silently_overridden(self) -> None:
        """A QAT choice cannot simultaneously select the dedicated MXFP8/fused variant."""
        for selectors in ({"use_mxfp8": True}, {"use_fused_mxfp8": True}):
            with self.subTest(selectors=selectors), self.assertRaises(ConfigResolutionError):
                family_config(use_low_precision=True, **selectors)

    def test_transposed_layout_preserves_loaded_values(self) -> None:
        """Noncanonical source weights are copied by the existing high-performance parent."""
        model = TinyMoE()
        before = model.mlp.experts.gate_up_proj.detach().clone()
        install_qat(model, qat_config())
        torch.testing.assert_close(model.mlp.experts.gate_up_proj, before.transpose(-2, -1))

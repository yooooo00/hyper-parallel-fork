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
"""Thin real-device launchers for state-aware attention Family defaults."""

import importlib.util

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("family", ["gdn", "kda"])
@pytest.mark.parametrize("mode", ["explicit", "default"])
def test_state_family_cp(family, mode):
    """
    Feature: Recurrent attention Family context parallel adapters.
    Description: Compare real CP2 state communication with the native CP1 layer.
    Expectation: Output, dX, all dW and default/explicit adapter contracts agree.
    """
    torchrun_case("tests/torch/context_parallel/_test_attention_family.py",
                  f"test_state_family_cp[{mode}-{family}]", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("mode", ["explicit", "default"])
def test_mla_family_cp(mode):
    """
    Feature: MLA Family context parallel adapter.
    Description: Preserve compressed latent projections, RoPE and selected kernel.
    Expectation: CP2 output, dX and dW match the CP1 baseline with distinct QK/V widths.
    """
    torchrun_case("tests/torch/context_parallel/_test_mla_family.py",
                  f"test_mla_family_cp[{mode}]", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("mode", ["explicit", "default"])
def test_dsa_family_cp(mode):
    """
    Feature: Omni-backed sparse attention Family context parallel adapter.
    Description: Compare real sparse indices and main attention gradients under CP2.
    Expectation: Matching global causal selection; explicitly skip when Omni is absent.
    """
    if importlib.util.find_spec("omni_training_custom_ops") is None:
        pytest.skip("DSA requires actual Omni sparse kernels; CPU mock is not numeric evidence")
    torchrun_case("tests/torch/context_parallel/_test_dsa_family.py",
                  f"test_dsa_family_cp[{mode}]", num_proc=2)

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
"""Lightweight launchers for real A5 grouped-expert family acceptance."""

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_single_device():
    """
    Feature: Non-fused MXFP8 grouped experts on A5.
    Description: Run numeric, lifecycle, empty and three-step optimizer checks.
    Expectation: Real float8 GMM calls and fixed numerical acceptance thresholds pass.
    """
    torchrun_case("tests/torch/expert_parallel/_test_grouped_experts_mxfp8.py", "test_single_device", num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("fused", [False, True])
def test_ep_two(fused):
    """
    Feature: Non-fused MXFP8 expert parallelism on two A5 devices.
    Description: Run planner-owned expert sharding with crossed and empty-rank routing.
    Expectation: Outputs and all gradients exactly match the order-aligned reference.
    """
    torchrun_case("tests/torch/expert_parallel/_test_grouped_experts_mxfp8.py", f"test_ep_two[{fused}]", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_fused_comparison():
    """
    Feature: Fused versus non-fused MXFP8 experts.
    Description: Compare output and gradients with identical groups and time training steps.
    Expectation: Predeclared numerical criteria pass with genuine fused forward/backward calls.
    """
    torchrun_case("tests/torch/expert_parallel/_test_grouped_experts_mxfp8.py", "test_fused_comparison", num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_ep_fused_comparison():
    """
    Feature: Fused and non-fused MXFP8 under EP=2.
    Description: Compare planner-sharded models with fixed weights, routes and gradients, including an empty rank.
    Expectation: Outputs and gradients meet the original thresholds and saved hooks receive physical tensors.
    """
    torchrun_case("tests/torch/expert_parallel/_test_grouped_experts_mxfp8.py", "test_ep_fused_comparison", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_three_way_loss():
    """
    Feature: BF16 and both MXFP8 expert loss trajectories.
    Description: Start with identical weights and run three SGD updates on fixed inputs and routing.
    Expectation: Losses and gradients are finite, outputs meet original tolerances, and fused kernels execute.
    """
    torchrun_case("tests/torch/expert_parallel/_test_grouped_experts_mxfp8.py", "test_three_way_loss", num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("case", ["test_fused_boundary", "test_fused_offload_checkpoint"])
def test_fused_lifecycle_and_boundary(case):
    """
    Feature: Fused operator boundaries and saved-tensor lifecycle on real A5.
    Description: Isolate activation quantization and exercise CPU offload plus recomputation.
    Expectation: Original numerical tolerances and exact lifecycle comparisons pass.
    """
    torchrun_case("tests/torch/expert_parallel/_test_grouped_experts_mxfp8.py", case, num_proc=1)

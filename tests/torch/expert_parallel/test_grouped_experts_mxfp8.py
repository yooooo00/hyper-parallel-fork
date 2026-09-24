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
def test_ep_two():
    """
    Feature: Non-fused MXFP8 expert parallelism on two A5 devices.
    Description: Run planner-owned expert sharding with crossed and empty-rank routing.
    Expectation: Outputs and all gradients exactly match the order-aligned reference.
    """
    torchrun_case("tests/torch/expert_parallel/_test_grouped_experts_mxfp8.py", "test_ep_two", num_proc=2)

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
"""Lightweight launchers for expanded four-rank grouped-expert acceptance."""

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("variant", ["baseline", "mxfp8", "fused"])
def test_ep_four(variant: str) -> None:
    """
    Feature: Grouped experts on four expert-parallel ranks.
    Description: Use the real planner and Qwen recipe with cross-rank and empty-rank routing.
    Expectation: Local expert shards, order, outputs and gradients match the reference.
    """
    torchrun_case("tests/torch/expert_parallel/_test_grouped_experts_ep_four.py",
                  f"test_ep_four[{variant}]", num_proc=4)

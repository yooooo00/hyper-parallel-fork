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
"""Lightweight launchers for actual A5 GQA projection validation."""

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("kind", ["llama", "qwen3_moe"])
@pytest.mark.parametrize("case", ["test_projection_numerics", "test_projection_saved_tensors",
                                 "test_projection_empty_input"])
def test_gqa_projections(kind: str, case: str) -> None:
    """
    Feature: GQA projection replacement on the actual device.
    Description: Run numerical, saved-tensor or empty-input cases for both source protocols.
    Expectation: Real MXFP8 kernels and lifecycle contracts satisfy fixed assertions.
    """
    torchrun_case("tests/torch/attention/_test_mxfp8_gqa_attention.py", f"{case}[{kind}]", num_proc=1)

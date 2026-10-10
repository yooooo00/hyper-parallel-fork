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
"""Lightweight launchers for the default NPU attention kernel under CP."""

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("kind", ["llama", "qwen3_moe"])
@pytest.mark.parametrize("low", [False, True])
def test_default_npu_cp(kind: str, low: bool) -> None:
    """
    Feature: GQA default NPU attention under CP.
    Description: Launch baseline or MXFP8 replacement on two ranks.
    Expectation: Outputs and gradients match the declared reference tolerances.
    """
    torchrun_case("tests/torch/attention/_test_gqa_npu_cp.py", f"test_default_npu_cp[{low}-{kind}]", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_default_npu_cp_quantized_reference() -> None:
    """
    Feature: MXFP8 CP communication semantics.
    Description: Compare against a reference using identical quantization partitions.
    Expectation: CP preserves expert-independent projection outputs and gradients.
    """
    torchrun_case("tests/torch/attention/_test_gqa_npu_cp.py", "test_default_npu_cp_quantized_reference", num_proc=2)

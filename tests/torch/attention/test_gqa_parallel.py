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
"""Framework-free launchers for the GQA Family TP/CP acceptance matrix."""

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("kind", ["llama", "qwen3_moe"])
@pytest.mark.parametrize("topology,adapter", [("tp", "default"), ("cp", "default"), ("cp", "explicit"),
                                            ("tp_cp", "default"), ("tp_cp", "explicit")])
def test_gqa_parallel(kind: str, topology: str, adapter: str) -> None:
    """
    Feature: GQA Family TP and CP composition.
    Description: Build both native attention protocols using real replacement and planning.
    Expectation: Baseline and MXFP8 outputs and gradients satisfy their fixed thresholds.
    """
    case = f"test_gqa_parallel[{adapter}-{topology}-{kind}]"
    torchrun_case("tests/torch/attention/_test_gqa_parallel.py", case,
                  num_proc=4 if topology == "tp_cp" else 2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("kind", ["llama", "qwen3_moe"])
def test_gqa_tp_validate(kind: str) -> None:
    """
    Feature: Unsupported validation-mode rejection.
    Description: Build GQA TP with validation dispatch enabled.
    Expectation: Construction fails with the attention module FQN.
    """
    torchrun_case("tests/torch/attention/_test_gqa_parallel.py", f"test_gqa_tp_validate[{kind}]", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("kind", ["llama", "qwen3_moe"])
def test_gqa_cp_masks(kind: str) -> None:
    """
    Feature: Global CP mask semantics.
    Description: Exercise default causal, padding and custom masks on two ranks.
    Expectation: Local query slices retain the same global key permissions.
    """
    torchrun_case("tests/torch/attention/_test_gqa_parallel.py", f"test_gqa_cp_masks[{kind}]", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_gqa_tp_bias() -> None:
    """
    Feature: Row-projection bias semantics under TP.
    Description: Compare baseline and MXFP8 projections with a nonzero output bias.
    Expectation: The output adds the bias once and all bias gradients match.
    """
    torchrun_case("tests/torch/attention/_test_gqa_parallel.py", "test_gqa_tp_bias", num_proc=2)

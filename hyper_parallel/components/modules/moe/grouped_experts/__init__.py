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
"""Public routed-expert configuration, family factory, and low-precision adapters."""

import importlib
from typing import TYPE_CHECKING, Any

from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import GroupedExpertsConfig
from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts_family import GroupedExperts

if TYPE_CHECKING:
    from hyper_parallel.components.modules.moe.grouped_experts.low_precision_grouped_experts import (
        FusedMXFP8GroupedSwiGLU,
        LowPrecisionGroupedExperts,
        MXFP8GroupedSwiGLU,
        QuantizedGroupedExperts,
        replace_fake_w4a8_grouped_experts,
        replace_grouped_experts_with_policy,
        replace_mxfp8_grouped_experts,
        replace_w4a8_grouped_experts,
    )

_LAZY_LOW_PRECISION_EXPORTS = frozenset({
    "FusedMXFP8GroupedSwiGLU",
    "LowPrecisionGroupedExperts",
    "MXFP8GroupedSwiGLU",
    "QuantizedGroupedExperts",
    "replace_fake_w4a8_grouped_experts",
    "replace_grouped_experts_with_policy",
    "replace_mxfp8_grouped_experts",
    "replace_w4a8_grouped_experts",
})

__all__ = [
    "FusedMXFP8GroupedSwiGLU",
    "GroupedExperts",
    "GroupedExpertsConfig",
    "LowPrecisionGroupedExperts",
    "MXFP8GroupedSwiGLU",
    "QuantizedGroupedExperts",
    "replace_fake_w4a8_grouped_experts",
    "replace_grouped_experts_with_policy",
    "replace_mxfp8_grouped_experts",
    "replace_w4a8_grouped_experts",
]


def __getattr__(name: str) -> Any:  # pylint: disable=invalid-name
    """Lazily load low-precision grouped-expert implementations."""
    if name not in _LAZY_LOW_PRECISION_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(
        "hyper_parallel.components.modules.moe.grouped_experts.low_precision_grouped_experts"
    )
    value = getattr(module, name)
    globals()[name] = value
    return value

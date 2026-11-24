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
"""Legacy policy-based replacement factories shared across models.

New configurations can select fixed strategies through Module Family rules.
These factories retain the original low-precision module layout and policy API.
"""

from collections.abc import Mapping
from typing import Any

from torch import nn

from hyper_parallel.components.modules.moe.grouped_experts.low_precision_grouped_experts import (
    QuantizedGroupedExperts,
    replace_grouped_experts_with_policy,
)
from hyper_parallel.components.quantization.functional import build_low_precision_strategy
from hyper_parallel.components.quantization.functional.linear_strategy_factory import (
    build_linear_strategy,
)
from hyper_parallel.components.quantization.modules.linear import LowPrecisionLinear
from hyper_parallel.models.replacement import module_replacement


@module_replacement
def replace_linear(
    *,
    module: nn.Module,
    module_fqn: str,
    context: Mapping[str, Any],
) -> LowPrecisionLinear:
    """Replace one exact Dense Linear using its resolved low-precision policy."""

    if type(module) is not nn.Linear:  # pylint: disable=unidiomatic-typecheck
        raise TypeError(
            f"{module_fqn!r} must be exact nn.Linear, got {type(module).__name__}"
        )
    if context.get("pp"):
        raise NotImplementedError(
            "Low-precision Linear training is not yet supported with pipeline parallelism."
        )
    try:
        strategy = build_linear_strategy(
            context.get("low_precision"),
            in_features=module.in_features,
            out_features=module.out_features,
        )
    except ValueError as error:
        raise ValueError(
            f"Low-precision Linear target {module_fqn!r} is invalid: {error}"
        ) from error
    return LowPrecisionLinear.from_linear(module, strategy=strategy)


@module_replacement
def replace_grouped_experts(
    *,
    module: nn.Module,
    module_fqn: str,
    context: Mapping[str, Any],
) -> QuantizedGroupedExperts:
    """Replace packed experts with the policy-selected grouped-linear strategy."""

    return replace_grouped_experts_with_policy(
        module=module,
        module_fqn=module_fqn,
        context=context,
        scheme=context.get("low_precision"),
        strategy_builder=build_low_precision_strategy,
    )


__all__ = ["replace_linear", "replace_grouped_experts"]

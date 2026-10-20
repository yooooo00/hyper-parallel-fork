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
"""Scheme-selected grouped experts using the existing QAT computation strategies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn
from torch.nn import functional as F

from hyper_parallel.components.modules.grouped_experts import GroupedExperts
from hyper_parallel.components.quantization.functional.base_gmm_func import _GroupedLinearFunction
from hyper_parallel.components.quantization.functional.strategy_factory import build_low_precision_strategy
from hyper_parallel.models.replacement import module_replacement


@module_replacement
class QATGroupedExperts(GroupedExperts):
    """Preserve high-performance expert layout/routing and bind one existing QAT strategy."""

    def __init__(self, *, module: nn.Module, module_fqn: str = "",
                 context: Mapping[str, Any] | None = None) -> None:
        """Build local expert computation from the already resolved dtype scheme.

        Args:
            module: Source packed experts.
            module_fqn: Module path used in diagnostics.
            context: Existing replacement context containing the resolved low_precision policy.
        """
        context = context or {}
        prefix = f"{module_fqn or '<root>'}: QATGroupedExperts"
        if any(context.get(axis) for axis in ("tp", "cp", "ep", "pp")) or context.get("ep_size", 1) != 1:
            raise NotImplementedError(f"{prefix} requires TP=CP=EP=PP=1")
        if context.get("low_precision") is None:
            raise ValueError(f"{prefix} requires a low_precision replacement entry and dtype scheme")
        super().__init__(module=module, module_fqn=module_fqn, context=context)
        activation = getattr(self.config, "hidden_act", "silu")
        if (not self.has_gate or self.add_bias or self.use_2d_experts
                or not self.router_gating_in_fp32 or activation not in ("silu", "swiglu")):
            raise ValueError(f"{prefix} requires 3D bias-free SwiGLU with routing scores after GMM2")
        self.grouped_linear = build_low_precision_strategy(
            context["low_precision"],
            tile_shapes=(tuple(self.gate_up_proj.transpose(-2, -1).shape),
                         tuple(self.down_proj.transpose(-2, -1).shape)),
        )

    def _grouped_gemm_expert_forward(self, gate_up_proj, down_proj, permuted,
                                     tokens_per_expert, permuted_probs):
        """Adapt layout at the shared strategy boundary; outer routing applies scores."""
        del permuted_probs
        projected = _GroupedLinearFunction.apply(
            permuted, gate_up_proj.transpose(-2, -1), tokens_per_expert, self.grouped_linear, 1)
        gate, up = projected.chunk(2, dim=-1)
        return _GroupedLinearFunction.apply(
            F.silu(gate) * up, down_proj.transpose(-2, -1), tokens_per_expert, self.grouped_linear, 1)

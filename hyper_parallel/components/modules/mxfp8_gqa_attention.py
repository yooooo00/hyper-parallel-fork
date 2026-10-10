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
"""MXFP8 projections with the ordinary grouped-query attention protocol."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.functional import npu_fusion_attention_forward
from hyper_parallel.components.modules.gqa_attention import GQAAttention
from hyper_parallel.components.quantization.functional.npu_mxfp8 import validate_npu_runtime
from hyper_parallel.components.quantization.modules.mxfp8_linear import MXFP8Linear
from hyper_parallel.models.replacement import module_replacement


def _convert_projection(projection: nn.Linear) -> MXFP8Linear:
    """Retain parameter identities and existing removable child-module hooks."""
    converted = MXFP8Linear.from_linear(projection)
    # Sharing the registries preserves previously returned RemovableHandle
    # objects, including the flags PyTorch uses for kwargs and full backward hooks.
    registries = (
        "_forward_pre_hooks", "_forward_pre_hooks_with_kwargs", "_forward_hooks",
        "_forward_hooks_with_kwargs", "_forward_hooks_always_called", "_backward_pre_hooks",
        "_backward_hooks", "_is_full_backward_hook", "_state_dict_hooks", "_state_dict_pre_hooks",
        "_load_state_dict_pre_hooks", "_load_state_dict_post_hooks",
    )
    for name in registries:
        setattr(converted, name, getattr(projection, name))
    return converted


@module_replacement
class MXFP8GQAAttention(GQAAttention):
    """Quantize QKV/output projections, leaving attention and communication BF16.

    The inherited forward retains child-module calls, normalization, rotary
    embeddings, cache handling and ``attention_interface``. Persistent QKV rows
    remain grouped by KV head; checkpoint transforms are inherited unchanged.
    """

    def __init__(
        self,
        *,
        module: nn.Module,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
        attention_interface: Callable[..., tuple[torch.Tensor, torch.Tensor | None]] = (
            npu_fusion_attention_forward
        ),
    ) -> None:
        """Build tile-aligned BF16 projections using existing MXFP8 Linear.

        Args:
            module: Source attention with standard fused or separate Q/K/V.
            module_fqn: Module path used in build-time capability errors.
            context: Parallel construction metadata.
            attention_interface: Ordinary attention implementation receiving BF16 Q/K/V.
        """
        prefix = f"{module_fqn or '<root>'}: MXFP8GQAAttention"
        if (context or {}).get("pp"):
            raise NotImplementedError(f"{prefix} does not support pipeline parallelism")
        try:
            super().__init__(module=module, module_fqn=module_fqn, context=context,
                             attention_interface=attention_interface)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{prefix}: {exc}") from exc
        for name in ("linear_qkv", "o_proj"):
            projection = getattr(self, name)
            if type(projection) is not nn.Linear:  # pylint: disable=unidiomatic-typecheck
                raise ValueError(f"{prefix}.{name} requires exact torch.nn.Linear semantics")
            if projection.weight.dtype != torch.bfloat16:
                raise ValueError(f"{prefix}.{name} requires BF16 parameters")
            if projection.bias is not None and (projection.bias.dtype != projection.weight.dtype
                                                or projection.bias.device != projection.weight.device):
                raise ValueError(f"{prefix}.{name} requires bias to share the weight dtype and device")
            if projection.in_features % 32 or projection.out_features % 32:
                raise ValueError(f"{prefix}.{name} requires input/output dimensions divisible by 32")
        try:
            validate_npu_runtime()
        except RuntimeError as exc:
            raise RuntimeError(f"{prefix}: {exc}") from exc
        self.linear_qkv = _convert_projection(self.linear_qkv)
        self.o_proj = _convert_projection(self.o_proj)


__all__ = ["MXFP8GQAAttention"]

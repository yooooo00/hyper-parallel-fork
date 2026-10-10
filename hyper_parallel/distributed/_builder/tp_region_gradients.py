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
"""Adjoints for regions consuming one replicated tensor across TP shards.

The output loss represents one logical replicated value, so an output SUM
has an identity backward. Local head computations contribute partial input
gradients, which the identity input boundary sums on backward. Generic
collectives retain their ordinary distributed autograd semantics.
"""

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.dtensor.placement_types import Partial, Placement, Replicate
from hyper_parallel.core.utils import communication


class _CopyToRegion(torch.autograd.Function):
    """Identity forward with a sum of the region's partial input gradients."""

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, group: Any) -> torch.Tensor:
        """Retain the group without saving an activation tensor.

        Args:
            ctx: Autograd context that retains the process group.
            tensor: Replicated input consumed by each model shard.
            group: Tensor-parallel process group."""
        ctx.group = group
        return tensor

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Sum disjoint model-shard contributions to the replicated input.

        Args:
            ctx: Autograd context carrying the tensor-parallel group.
            gradient: This model shard's contribution to the input gradient."""
        return communication.differentiable_all_reduce(gradient, "sum", ctx.group), None


class _ReduceFromRegion(torch.autograd.Function):
    """Sum forward contributions with an identity replicated-output gradient."""

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, group: Any) -> torch.Tensor:
        """Reduce model-shard contributions into the logical replicated output.

        Args:
            ctx: Unused autograd context; no activation is saved.
            tensor: Partial output computed by this model shard.
            group: Tensor-parallel process group."""
        del ctx
        return communication.differentiable_all_reduce(tensor, "sum", group)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Each model shard consumes the same single logical output gradient.

        Args:
            ctx: Unused autograd context.
            gradient: Gradient of the replicated logical output."""
        del ctx
        return gradient, None


@dataclass(frozen=True)
class TPRegionGradientOp:
    """One explicit placement transition usable in either execution mode."""

    group: Any
    input_side: bool
    preserves_layout: bool = True

    def execute(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the correct replicated-value adjoint to a local tensor.

        Args:
            tensor: Local tensor at the region's input or output boundary."""
        operation = _CopyToRegion if self.input_side else _ReduceFromRegion
        return operation.apply(tensor, self.group)


def region_gradient_operation(
    mesh: Any, mesh_dim_names: Sequence[str], src: Sequence[Placement],
    dst: Sequence[Placement], *, input_side: bool,
) -> Optional[TPRegionGradientOp]:
    """Lower only supported TP transitions; SP retains its existing adjoint.

    Args:
        mesh: Active mesh whose order matches the placement declarations.
        mesh_dim_names: Axis names for those declarations.
        src: Source placements before entering or exiting the region.
        dst: Target placements after the transition.
        input_side: Whether this is the entry to a partial-compute region.

    Returns:
        A layout-preserving operation, or None for unrelated transitions.
    """
    if mesh is None or "tp" not in mesh_dim_names:
        return None
    index = tuple(mesh_dim_names).index("tp")
    if any(left != right for axis, (left, right) in enumerate(zip(src, dst)) if axis != index):
        return None
    if input_side:
        selected = src == dst and isinstance(src[index], Replicate)
    else:
        selected = (isinstance(src[index], Partial) and src[index].reduce_op == "sum"
                    and isinstance(dst[index], Replicate))
    if not selected:
        return None
    tp_mesh = mesh if tuple(mesh_dim_names) == ("tp",) else mesh["tp"]
    return TPRegionGradientOp(tp_mesh.get_group(), input_side)

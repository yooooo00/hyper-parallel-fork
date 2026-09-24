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
"""Preflight the narrow non-fused GroupedExperts/MXFP8 parallel contract."""

from typing import Any

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard


def _validate_mesh(module, mesh, expert_mesh, validate_mode, prefix):
    """Validate effective sizes, not just replacement context flags."""
    if validate_mode:
        raise NotImplementedError(f"{prefix} supports production mode only")
    for axis in ("tp", "cp", "pp"):
        if axis in mesh.mesh_dim_names and mesh[axis].size() > 1:
            raise NotImplementedError(f"{prefix} requires TP=CP=PP=1; active {axis}")
    ep_size = 1 if expert_mesh is None else expert_mesh["ep"].size()
    if ep_size not in (1, 2) or module.num_experts % ep_size:
        raise NotImplementedError(f"{prefix} requires EP=1 or 2 dividing the expert count")
    return ep_size


def _validate_placements(spec, placements, ep_size, prefix):
    """Only the existing planner's expert-axis sharding can partition these weights."""
    has_ep_shard = False
    for axis, placement in placements.items():
        axis = getattr(axis, "value", axis)
        if isinstance(placement, Replicate):
            continue
        if axis != "ep" or not isinstance(placement, Shard) or placement.dim != 0:
            raise NotImplementedError(f"{prefix}: unsupported placement {axis}={placement}")
        has_ep_shard = True
    if ep_size == 2 and (not has_ep_shard or spec._ep_size != 2):
        raise ValueError(f"{prefix}: EP=2 requires planner-owned Shard(0) on both expert weights")
    if ep_size == 2 and spec.local_compute_fn is None:
        raise ValueError(f"{prefix}: EP requires an explicit model recipe local_compute_fn")
    compute = spec.local_compute_fn
    if hasattr(compute, "to_dict") and compute.to_dict().get("use_grouped_gemm") is False:
        raise ValueError(f"{prefix}: explicit use_grouped_gemm=False is unsupported")


def _validate_parameter_plan(fqn, plan, ep_size, prefix):
    """Find each canonical weight in the final plan, including user overrides."""
    for name in ("gate_up_proj", "down_proj"):
        parameter_fqn = f"{fqn}.{name}" if fqn else name
        entries = [(spec, placements) for boundary, spec in plan.modules.items()
                   for param, placements in spec.params.items()
                   if (f"{boundary}.{param}" if boundary else param) == parameter_fqn]
        if ep_size == 2 and len(entries) != 1:
            raise ValueError(f"{prefix}: plan must shard {parameter_fqn} exactly once")
        for spec, placements in entries:
            _validate_placements(spec, placements, ep_size, prefix)


def validate_module_family_plan(model: Any, plan: Any, mesh: Any,
                                expert_mesh: Any, *, validate_mode: bool) -> None:
    """Reject unsupported meshes, execution modes and expert placements before sharding.

    Args:
        model: Model containing the final replacement modules.
        plan: Final planner output, including explicit user overrides.
        mesh: Full dense mesh, including size-one axes.
        expert_mesh: Effective expert mesh derived by the existing builder.
        validate_mode: Whether parameters would remain DTensors during execution.
    """
    candidates = [(fqn, module) for fqn, module in model.named_modules()
                  if getattr(module, "requires_grouped_expert_compute", False)]
    if not candidates:
        return
    # Keep the optional low-precision implementation out of baseline-only builds.
    from hyper_parallel.components.modules.mxfp8_grouped_swiglu import (  # pylint: disable=C0415
        MXFP8GroupedSwiGLU,
    )
    for fqn, module in candidates:
        if isinstance(module, MXFP8GroupedSwiGLU):
            prefix = f"{fqn}: MXFP8GroupedSwiGLU"
            ep_size = _validate_mesh(module, mesh, expert_mesh, validate_mode, prefix)
            _validate_parameter_plan(fqn, plan, ep_size, prefix)

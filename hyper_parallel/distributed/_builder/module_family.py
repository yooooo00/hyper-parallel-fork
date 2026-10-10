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
"""Family defaults, build summaries and directed parallel capability checks."""

from typing import Any

from hyper_parallel.components.modules.family import (
    get_module_family_selection,
    import_family_symbol,
)
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard


def apply_module_family_defaults(model: Any, plan: Any, *, cp_size: int,
                                 source_gradient_reduction: bool = False) -> None:
    """Fill undeclared Family contracts after override boundaries have been inserted.

    Args:
        model: Model whose replacements have already been constructed.
        plan: Mutable plan containing derived and explicitly inserted boundaries.
        cp_size: Effective context-parallel group size.
        source_gradient_reduction: Whether HSDP owns source-layout gradient
            reduction, disabling the standalone replicated-value TP adjoint.

    Raises:
        ValueError: A Family requiring CP has no declared planner boundary.
    """
    for fqn, module in model.named_modules():
        selection = get_module_family_selection(module)
        if selection is None:
            continue
        family = selection.family
        spec = plan.modules.get(fqn)
        if spec is not None and family.tp_replicated_gradient_contract:
            spec._tp_replicated_gradient_contract = not source_gradient_reduction
        default = family.parallel_defaults.get("cp")
        if default is not None and cp_size > 1:
            if spec is None:
                raise ValueError(f"{fqn}: Family {family.name} CP requires a planner boundary for this module")
            explicitly_disabled = spec.inner_wrapper is None and spec._family_adapter_source == "explicit"
            if not explicitly_disabled:
                if spec.inner_wrapper is None:
                    spec.inner_wrapper = import_family_symbol(default)
                    spec._family_adapter_source = "family default"
                if spec.inner_target is None:
                    spec.inner_target = "self"
                if spec.region_dispatch is None:
                    spec.region_dispatch = False
        plan.family_modules[fqn] = {
            "family": family.name, "variant": selection.variant,
            "implementation": selection.implementation,
            "parameter_layout": family.parameter_layout,
            "tp_replicated_gradient_contract": bool(spec and spec._tp_replicated_gradient_contract),
            "tp_gradient_contract": "source_layout" if source_gradient_reduction else "replicated",
        }


def finalize_module_family_summary(model: Any, plan: Any, *, tp_size: int, cp_size: int, ep_size: int) -> None:
    """Expose final adapter selection and reject explicitly disabled required adapters.

    Args:
        model: Model containing the final selected Family implementations.
        plan: Mutable plan whose Family defaults and explicit choices are resolved.
        tp_size: Effective dense tensor-parallel size.
        cp_size: Effective context-parallel size.
        ep_size: Effective expert-parallel size.

    Raises:
        ValueError: An active Family CP protocol has been explicitly disabled.
    """
    modules = dict(model.named_modules())
    for fqn, summary in plan.family_modules.items():
        spec = plan.modules.get(fqn)
        selection = get_module_family_selection(modules[fqn])
        wrapper = spec.inner_wrapper if spec is not None else None
        if cp_size > 1 and selection.family.parallel_defaults.get("cp") is not None and wrapper is None:
            raise ValueError(f"{fqn}: Family {selection.family.name} requires a CP adapter; explicitly disabled")
        summary.update(
            adapter=plan._format_injection_callable(wrapper),
            adapter_source=getattr(spec, "_family_adapter_source", None) or ("explicit" if wrapper else "not required"),
            parallel_sizes={"tp": tp_size, "cp": cp_size, "ep": ep_size},
        )


def _axis_size(mesh: Any, axis: str) -> int:
    return mesh[axis].size() if mesh is not None and axis in mesh.mesh_dim_names else 1


def validate_attention_protocol_plan(fqn: str, module: Any, selection: Any, plan: Any, mesh: Any,
                                     expert_mesh: Any, *, validate_mode: bool) -> None:
    """Retain the separately validated CP-only contracts for non-GQA attention.

    Args:
        fqn: Full module path included in capability errors.
        module: Final module instance, unused by this topology-only check.
        selection: Registered Family and selected implementation metadata.
        plan: Final plan, unused by this topology-only check.
        mesh: Dense runtime mesh carrying TP and PP dimensions.
        expert_mesh: Expert mesh, unused by attention.
        validate_mode: DTensor validation switch, unused by this topology check.

    Raises:
        NotImplementedError: Tensor or pipeline parallelism exceeds one.
    """
    del module, plan, expert_mesh, validate_mode
    for axis in ("tp", "pp"):
        if _axis_size(mesh, axis) > 1:
            raise NotImplementedError(f"{fqn}: Family {selection.family.name} requires {axis.upper()}=1")


def validate_gqa_plan(fqn: str, module: Any, selection: Any, plan: Any, mesh: Any,
                      expert_mesh: Any, *, validate_mode: bool) -> None:
    """Validate final local projection dimensions, not merely global head sizes.

    Args:
        fqn: Full attention module path included in capability errors.
        module: Selected GQA module containing persistent projection weights.
        selection: Registered GQA Family and selected precision variant.
        plan: Final parameter placements across parent and child boundaries.
        mesh: Dense runtime mesh used to derive each projection's local shape.
        expert_mesh: Expert mesh, unused by GQA projections.
        validate_mode: Whether execution requires DTensor operator dispatch.

    Raises:
        ValueError: A projection is missing, has an invalid TP role, or is unaligned.
        NotImplementedError: Requested dispatch or pipeline mode is unsupported.
    """
    del expert_mesh
    prefix = f"{fqn}: Family {selection.family.name} ({selection.variant})"
    if selection.variant == "baseline":
        if validate_mode and _axis_size(mesh, "tp") > 1:
            raise NotImplementedError(
                f"{prefix}: TP validate mode requires RoPE DTensor dispatch with mixed tensor position embeddings"
            )
        return
    if validate_mode:
        raise NotImplementedError(f"{prefix}: MXFP8 DTensor operator dispatch is not supported in validate mode")
    if _axis_size(mesh, "pp") > 1:
        raise NotImplementedError(f"{prefix}: PP is not supported")
    if fqn not in plan.modules:
        raise ValueError(f"{prefix}: requires a planner-owned attention boundary")
    for name, shard_dimension in (("linear_qkv.weight", 0), ("o_proj.weight", 1)):
        placements = _gqa_projection_placements(fqn, name, plan, prefix)
        shape = _gqa_local_projection_shape(module.get_parameter(name).shape, placements,
                                             mesh, shard_dimension, f"{prefix}.{name}")
        if any(dimension % 32 for dimension in shape):
            raise ValueError(f"{prefix}: local {name} shape {tuple(shape)} must be aligned to 32")


def _gqa_projection_placements(fqn: str, parameter: str, plan: Any, prefix: str) -> dict:
    """Resolve a parameter once across parent and child planner boundaries."""
    parameter_fqn = f"{fqn}.{parameter}" if fqn else parameter
    entries = [placements for boundary, spec in plan.modules.items()
               for name, placements in spec.params.items()
               if (f"{boundary}.{name}" if boundary else name) == parameter_fqn]
    if len(entries) != 1:
        raise ValueError(f"{prefix}: plan must declare {parameter_fqn} exactly once, got {len(entries)}")
    return entries[0]


def _gqa_local_projection_shape(shape: Any, placements: dict, mesh: Any,
                                shard_dimension: int, prefix: str) -> list[int]:
    """Retain QKV column/output row TP roles while checking local tile sizes."""
    local_shape = list(shape)
    tp_size = _axis_size(mesh, "tp")
    has_tp_shard = False
    for axis, placement in placements.items():
        axis = getattr(axis, "value", axis)
        if isinstance(placement, Replicate):
            continue
        if (axis != "tp" or not isinstance(placement, Shard)
                or placement.dim not in (shard_dimension, shard_dimension - len(local_shape))):
            raise ValueError(f"{prefix}: unsupported projection placement {axis}={placement}; "
                             f"requires TP Shard({shard_dimension})")
        has_tp_shard = True
        if local_shape[shard_dimension] % tp_size:
            raise ValueError(f"{prefix}: cannot be divided across TP={tp_size}")
        local_shape[shard_dimension] //= tp_size
    if tp_size > 1 and not has_tp_shard:
        raise ValueError(f"{prefix}: TP={tp_size} requires TP Shard({shard_dimension})")
    return local_shape


def validate_experts_plan(fqn: str, module: Any, selection: Any, plan: Any, mesh: Any,
                          expert_mesh: Any, *, validate_mode: bool) -> None:
    """Dispatch the declared low-precision expert contract without class-name inference.

    Args:
        fqn: Full expert module path included in capability errors.
        module: Selected expert module with its global expert count.
        selection: Registered Family and selected precision variant.
        plan: Final expert parameter placements and local-compute recipe.
        mesh: Dense runtime mesh used for pipeline capability checks.
        expert_mesh: Planner-derived expert mesh with an EP axis, or None.
        validate_mode: Whether execution requires DTensor operator dispatch.

    Raises:
        ValueError: Expert weights or the grouped recipe violate the EP contract.
        NotImplementedError: The topology, dispatch mode, or placement is unsupported.
    """
    if selection.variant == "baseline":
        return
    prefix = f"{fqn}: Family {selection.family.name} ({selection.variant})"
    ep_size = _validate_mesh(module, mesh, expert_mesh, validate_mode, prefix)
    _validate_parameter_plan(fqn, plan, ep_size, prefix)


def validate_module_family_plan(model: Any, plan: Any, mesh: Any,
                                expert_mesh: Any, *, validate_mode: bool) -> None:
    """Run only the validators declared by each final module's exact Family.

    Args:
        model: Model containing the final replacements to validate.
        plan: Fully resolved sharding plan after explicit overrides.
        mesh: Dense runtime mesh.
        expert_mesh: Planner-derived expert mesh, or None when EP is inactive.
        validate_mode: Whether execution requires DTensor operator dispatch.
    """
    for fqn, module in model.named_modules():
        selection = get_module_family_selection(module)
        if selection is None or selection.family.plan_validator is None:
            continue
        validate = import_family_symbol(selection.family.plan_validator)
        validate(fqn, module, selection, plan, mesh, expert_mesh, validate_mode=validate_mode)


def _validate_mesh(module, mesh, expert_mesh, validate_mode, prefix):
    """Validate effective sizes, not just replacement context flags."""
    if validate_mode:
        raise NotImplementedError(f"{prefix} supports production mode only")
    if _axis_size(mesh, "pp") > 1:
        raise NotImplementedError(f"{prefix} requires PP=1")
    ep_size = 1 if expert_mesh is None else expert_mesh["ep"].size()
    if ep_size not in (1, 2, 4) or module.num_experts % ep_size:
        raise NotImplementedError(f"{prefix} requires EP=1, 2 or 4 dividing the expert count")
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
    if ep_size > 1 and (not has_ep_shard or spec._ep_size != ep_size):
        raise ValueError(f"{prefix}: EP>1 requires planner-owned Shard(0) on both expert weights")
    if ep_size > 1 and spec.local_compute_fn is None:
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
        if ep_size > 1 and len(entries) != 1:
            raise ValueError(f"{prefix}: plan must shard {parameter_fqn} exactly once")
        for spec, placements in entries:
            _validate_placements(spec, placements, ep_size, prefix)

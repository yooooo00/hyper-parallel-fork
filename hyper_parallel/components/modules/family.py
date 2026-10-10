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
"""Exact module families and their construction-time implementation contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from inspect import Parameter, signature
from typing import Any, Callable, Mapping


def _symbol_path(symbol: Callable | str) -> str:
    """Return a stable import identity without importing optional implementations."""
    if isinstance(symbol, str):
        return symbol
    return f"{symbol.__module__}.{symbol.__qualname__}"


def import_family_symbol(path: str) -> Any:
    """Import a declared implementation or adapter only when it is requested.

    Args:
        path: Fully qualified import path from an explicit Family declaration.
    """
    module_path, _, name = path.rpartition(".")
    return getattr(import_module(module_path), name)


@dataclass(frozen=True)
class ModuleFamily:
    """Exact baseline, optional variants, parallel defaults and plan validator.

    Import paths keep optional device kernels out of baseline configuration parsing.
    ``parallel_defaults`` maps an axis to an ``@inner_wrapper`` factory path.
    Subclasses require their own explicit registration.
    """

    name: str
    baseline: Callable | str
    variants: Mapping[str, str] = field(default_factory=dict)
    parallel_defaults: Mapping[str, str] = field(default_factory=dict)
    plan_validator: str | None = None
    parameter_layout: str = ""
    tp_replicated_gradient_contract: bool = False


@dataclass(frozen=True)
class FamilySelection:
    """Construction result carried by the final module through planning."""

    family: ModuleFamily
    variant: str
    implementation: str


_FAMILIES: dict[str, ModuleFamily] = {}


def register_module_family(family: ModuleFamily) -> None:
    """Register one exact baseline; subclasses do not inherit membership.

    Args:
        family: Baseline identity and explicitly supported import paths.
    """
    path = _symbol_path(family.baseline)
    if path in _FAMILIES or any(item.name == family.name for item in _FAMILIES.values()):
        raise ValueError(f"Module family already registered: {family.name}")
    _FAMILIES[path] = family


def get_module_family(target: Callable) -> ModuleFamily | None:
    """Return the family of an exact baseline callable, without subclass inference.

    Args:
        target: Baseline constructor whose identity is checked against registration.
    """
    family = _FAMILIES.get(_symbol_path(target))
    if family is None:
        return None
    baseline = import_family_symbol(family.baseline) if isinstance(family.baseline, str) else family.baseline
    return family if baseline is target else None


def get_module_family_selection(module: Any) -> FamilySelection | None:
    """Read the construction result or recognize an explicitly registered exact type.

    Args:
        module: Final module instance to inspect without changing its state.
    """
    selection = vars(module).get("_hp_family_selection")
    if selection is not None:
        return selection
    path = _symbol_path(type(module))
    family = get_module_family(type(module))
    if family is not None:
        return FamilySelection(family, "baseline", path)
    for family in _FAMILIES.values():
        for variant, implementation in family.variants.items():
            if path == implementation and import_family_symbol(implementation) is type(module):
                return FamilySelection(family, variant, path)
    return None


def bind_module_family(module: Any, baseline: Callable, selectors: Mapping[str, bool]) -> None:
    """Attach the selected Family to a constructed module without registering state.

    Args:
        module: Replacement result receiving non-persistent construction metadata.
        baseline: Original public replacement target.
        selectors: Validated precision selectors used to construct this instance.
    """
    family = get_module_family(baseline)
    if family is None:
        return
    variant = "baseline"
    if selectors.get("use_mxfp8", False):
        variant = "fused_mxfp8" if selectors.get("use_fused_mxfp8", False) else "mxfp8"
    module._hp_family_selection = FamilySelection(family, variant, _symbol_path(type(module)))


def _validate_precision_selectors(selectors: Mapping[str, bool], prefix: str) -> None:
    """Check selector types and prerequisites before choosing an implementation."""
    if any(not isinstance(value, bool) for value in selectors.values()):
        raise TypeError(f"{prefix}: ModuleFamily selectors must be bool: {selectors}")
    if selectors.get("use_fused_mxfp8", False) and not selectors.get("use_mxfp8", False):
        raise ValueError(f"{prefix}: use_fused_mxfp8 requires use_mxfp8=True")


def resolve_module_family_target(target: Callable, arguments: Mapping) -> tuple[Callable, dict]:
    """Consume selectors, select an implementation and strictly validate its keywords.

    Args:
        target: Exact baseline or a non-family replacement callable.
        arguments: Selectors and constructor arguments; never modified in place.

    Returns:
        Concrete callable and constructor arguments without selection keys.
    """
    args = dict(arguments)
    family = get_module_family(target)
    selectors = {key: args.pop(key) for key in ("use_mxfp8", "use_fused_mxfp8") if key in args}
    prefix = f"{args.get('module_fqn', '<configuration>')}: Family {family.name if family else _symbol_path(target)}"
    if selectors and family is None:
        raise ValueError(f"{prefix}: low-precision selectors require a registered ModuleFamily")
    _validate_precision_selectors(selectors, prefix)
    if selectors.get("use_mxfp8", False):
        variant = "fused_mxfp8" if selectors.get("use_fused_mxfp8") else "mxfp8"
        implementation = family.variants.get(variant)
        if implementation is None:
            raise ValueError(f"{prefix}: unsupported selectors {selectors}; variant {variant!r} is unavailable")
        target = import_family_symbol(implementation)
    parameters = signature(target).parameters
    unknown = set(args) - {key for key, value in parameters.items()
                           if value.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)}
    if unknown:
        raise TypeError(f"{prefix}: {target.__name__}: unknown replacement arguments: {sorted(unknown)}")
    signature(target).bind_partial(**args)
    return target, args


register_module_family(ModuleFamily(
    name="grouped_experts",
    baseline="hyper_parallel.components.modules.grouped_experts.GroupedExperts",
    variants={
        "mxfp8": "hyper_parallel.components.modules.mxfp8_grouped_swiglu.MXFP8GroupedSwiGLU",
        "fused_mxfp8": "hyper_parallel.components.modules.mxfp8_grouped_swiglu.FusedMXFP8GroupedSwiGLU",
    },
    plan_validator="hyper_parallel.distributed._builder.module_family.validate_experts_plan",
    parameter_layout="gate_up_proj=[E,H,2I]; down_proj=[E,I,H]",
))
register_module_family(ModuleFamily(
    name="gqa_attention",
    tp_replicated_gradient_contract=True,
    baseline="hyper_parallel.components.modules.gqa_attention.GQAAttention",
    variants={"mxfp8": "hyper_parallel.components.modules.mxfp8_gqa_attention.MXFP8GQAAttention"},
    parallel_defaults={
        "cp": "hyper_parallel.distributed.context_parallel.attention_interface.attention_interface_cp_wrapper",
    },
    plan_validator="hyper_parallel.distributed._builder.module_family.validate_gqa_plan",
    parameter_layout="linear_qkv=[KV_heads*(Q_per_KV*Dq+Dk+Dv),H]; o_proj=[H,Q_heads*Dv]",
))

for _name, _baseline, _adapter, _layout in (
    ("mla_attention", "hyper_parallel.components.modules.mla_attention.MLAAttention", "mla.mla_attention_cp_wrapper",
     "linear_qkv=concat(q_a,kv_a); q_b/kv_b/o_proj retain source layout; CP parameters replicated"),
    ("dsa_attention", "hyper_parallel.components.modules.dsa_attention.DeepseekV32DSAAttention",
     "dsa.dsa_attention_cp_wrapper",
     "packed latent Q/KV and indexer parameters retain replacement layout; CP parameters replicated"),
    ("gated_delta_net", "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet",
     "state.gated_delta_net_cp_wrapper", "native projection/convolution/state parameter layout; CP replicated"),
    ("kimi_delta_attention", "hyper_parallel.components.modules.kimi_delta_attention.KimiDeltaAttention",
     "state.kimi_delta_attention_cp_wrapper", "native projection/convolution/state parameter layout; CP replicated"),
):
    register_module_family(ModuleFamily(
        name=_name, baseline=_baseline,
        parallel_defaults={"cp": f"hyper_parallel.distributed.attention_family.{_adapter}"},
        plan_validator="hyper_parallel.distributed._builder.module_family.validate_attention_protocol_plan",
        parameter_layout=_layout,
    ))

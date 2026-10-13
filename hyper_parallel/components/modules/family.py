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
"""Construction-time module families, keyed by exact baseline class identity."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from inspect import Parameter, signature
from typing import Any, Callable, Mapping

from hyper_parallel.components.modules.grouped_experts import GroupedExperts


@dataclass(frozen=True)
class SelectorSpec:
    """The exact Python type and baseline default of one construction selector."""

    value_type: type
    default: Any


@dataclass(frozen=True)
class VariantRule:
    """A complete named selector combination pointing to a declared variant."""

    when: Mapping[str, Any]
    variant: str


@dataclass(frozen=True)
class ModuleFamily:
    """A baseline and its explicitly supported, lazily imported variants."""

    name: str
    baseline: Callable
    variants: Mapping[str, str]
    selector_specs: Mapping[str, SelectorSpec] = field(default_factory=dict)
    variant_rules: tuple[VariantRule, ...] = ()


_FAMILIES: dict[Callable, ModuleFamily] = {}


def _validate_selector_value(family: ModuleFamily, name: str, value: Any) -> None:
    """Keep selector types exact, including the distinction between bool and int."""
    expected = family.selector_specs[name].value_type
    actual_type = type(value)
    if actual_type is not expected:
        raise TypeError(f"ModuleFamily {family.name}: selector {name!r} must be {expected.__name__}, "
                        f"got {actual_type.__name__}: {value!r}")


def _validate_family_rules(family: ModuleFamily) -> None:
    """Reject incomplete, ambiguous or unreachable declarations before selection."""
    defaults = {}
    for name, spec in family.selector_specs.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"ModuleFamily {family.name}: selector names must be non-empty strings")
        if not isinstance(spec, SelectorSpec) or not isinstance(spec.value_type, type):
            raise TypeError(f"ModuleFamily {family.name}: {name!r} requires a SelectorSpec with a type")
        _validate_selector_value(family, name, spec.default)
        defaults[name] = spec.default
    conditions = [defaults]
    for rule in family.variant_rules:
        if set(rule.when) != set(defaults):
            raise ValueError(f"ModuleFamily {family.name}: rule {rule.when!r} must contain exactly "
                             f"the selector keys {sorted(defaults)}")
        for name, value in rule.when.items():
            _validate_selector_value(family, name, value)
        if rule.variant not in family.variants:
            raise ValueError(f"ModuleFamily {family.name}: undeclared variant {rule.variant!r}")
        condition = dict(rule.when)
        if condition in conditions:
            raise ValueError(f"ModuleFamily {family.name}: duplicate or baseline-default rule {condition!r}")
        conditions.append(condition)


def register_module_family(family: ModuleFamily) -> None:
    """Register one exact baseline; subclasses do not inherit family membership.

    Args:
        family: Baseline identity and explicitly supported variant import paths.
    """
    if not callable(family.baseline):
        raise TypeError(f"ModuleFamily {family.name}: baseline must be callable")
    if family.baseline in _FAMILIES:
        raise ValueError(f"Module family already registered: {family.name}")
    _validate_family_rules(family)
    _FAMILIES[family.baseline] = family


def get_module_family(target: Callable) -> ModuleFamily | None:
    """Return the family registered for this exact callable.

    Args:
        target: Requested replacement callable, compared by identity.
    """
    return _FAMILIES.get(target)


def get_family_selector_specs(target: Callable) -> Mapping[str, SelectorSpec]:
    """Return the selector schema for an exact registered baseline.

    Args:
        target: Requested baseline or an ordinary replacement callable.

    Returns:
        The registered selector declarations, or an empty mapping for non-families.
    """
    family = get_module_family(target)
    return family.selector_specs if family is not None else {}


def _select_variant(family: ModuleFamily, args: dict) -> Callable:
    """Consume declared selectors and resolve their complete named combination."""
    selectors = {name: args.pop(name, spec.default) for name, spec in family.selector_specs.items()}
    for name, value in selectors.items():
        _validate_selector_value(family, name, value)
    defaults = {name: spec.default for name, spec in family.selector_specs.items()}
    if selectors == defaults:
        return family.baseline
    matches = [rule for rule in family.variant_rules if selectors == dict(rule.when)]
    if len(matches) != 1:
        legal = [defaults, *(dict(rule.when) for rule in family.variant_rules)]
        raise ValueError(f"ModuleFamily {family.name}: unsupported selector combination {selectors}; "
                         f"legal combinations: {legal}")
    module_path, _, name = family.variants[matches[0].variant].rpartition(".")
    return getattr(import_module(module_path), name)


def resolve_module_family_target(target: Callable, arguments: Mapping) -> tuple[Callable, dict]:
    """Consume selectors, choose a concrete implementation, and validate arguments.

    Unknown keywords are rejected even for a callable accepting **kwargs.
    The input mapping is never modified. Only GroupedExperts is registered.

    Args:
        target: Requested baseline or non-family replacement callable.
        arguments: Configured selectors and constructor arguments.

    Returns:
        Concrete callable and constructor arguments with selectors removed.
    """
    args = dict(arguments)
    family = get_module_family(target)
    if family is not None:
        target = _select_variant(family, args)
    else:
        selector_names = {name for registered in _FAMILIES.values() for name in registered.selector_specs}
        selectors = set(args) & selector_names
        if selectors:
            raise ValueError(f"{target}: selectors {sorted(selectors)} require a registered ModuleFamily")
    parameters = signature(target).parameters
    unknown = set(args) - {key for key, value in parameters.items()
                           if value.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)}
    if unknown:
        raise TypeError(f"{target.__name__}: unknown replacement arguments: {sorted(unknown)}")
    signature(target).bind_partial(**args)
    return target, args


register_module_family(ModuleFamily(
    name="grouped_experts",
    baseline=GroupedExperts,
    variants={
        "mxfp8": "hyper_parallel.components.modules.mxfp8_grouped_swiglu.MXFP8GroupedSwiGLU",
        "fused_mxfp8": "hyper_parallel.components.modules.mxfp8_grouped_swiglu.FusedMXFP8GroupedSwiGLU",
    },
    selector_specs={
        "use_mxfp8": SelectorSpec(bool, False),
        "use_fused_mxfp8": SelectorSpec(bool, False),
    },
    variant_rules=(
        VariantRule({"use_mxfp8": True, "use_fused_mxfp8": False}, "mxfp8"),
        VariantRule({"use_mxfp8": True, "use_fused_mxfp8": True}, "fused_mxfp8"),
    ),
))

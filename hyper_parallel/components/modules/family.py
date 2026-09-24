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

from dataclasses import dataclass
from importlib import import_module
from inspect import Parameter, signature
from typing import Callable, Mapping

from hyper_parallel.components.modules.grouped_experts import GroupedExperts


@dataclass(frozen=True)
class ModuleFamily:
    """A baseline and its explicitly supported, lazily imported variants."""

    name: str
    baseline: Callable
    variants: Mapping[str, str]


_FAMILIES: dict[Callable, ModuleFamily] = {}


def register_module_family(family: ModuleFamily) -> None:
    """Register one exact baseline; subclasses do not inherit family membership.

    Args:
        family: Baseline identity and explicitly supported variant import paths.
    """
    if family.baseline in _FAMILIES:
        raise ValueError(f"Module family already registered: {family.name}")
    _FAMILIES[family.baseline] = family


def get_module_family(target: Callable) -> ModuleFamily | None:
    """Return the family registered for this exact callable.

    Args:
        target: Requested replacement callable, compared by identity.
    """
    return _FAMILIES.get(target)


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
    selectors = {key: args.pop(key) for key in ("use_mxfp8", "use_fused_mxfp8") if key in args}
    if selectors and family is None:
        raise ValueError(f"{target}: low-precision selectors require a registered ModuleFamily")
    if any(not isinstance(value, bool) for value in selectors.values()):
        raise TypeError("ModuleFamily selectors must be bool")
    if selectors.get("use_fused_mxfp8", False):
        if not selectors.get("use_mxfp8", False):
            raise ValueError("use_fused_mxfp8 requires use_mxfp8=True")
        raise NotImplementedError("use_fused_mxfp8=True is not supported; use non-fused MXFP8")
    if selectors.get("use_mxfp8", False):
        module_path, _, name = family.variants["mxfp8"].rpartition(".")
        target = getattr(import_module(module_path), name)
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
    variants={"mxfp8": "hyper_parallel.components.modules.mxfp8_grouped_swiglu.MXFP8GroupedSwiGLU"},
))

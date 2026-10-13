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
"""Declarative selectors through real configuration and replacement entry points."""

from dataclasses import replace
from importlib import import_module
import unittest
from unittest.mock import PropertyMock, patch

import torch
import yaml

from hyper_parallel.components.modules import family as family_registry
from hyper_parallel.components.modules.family import (
    ModuleFamily, SelectorSpec, VariantRule, get_module_family,
    register_module_family, resolve_module_family_target,
)
from hyper_parallel.components.modules.grouped_experts import GroupedExperts
from hyper_parallel.components.modules.mxfp8_grouped_swiglu import (
    FusedMXFP8GroupedSwiGLU, MXFP8GroupedSwiGLU,
)
from hyper_parallel.components.quantization.quantizers.mxfp8 import MXFP8Quantizer
from hyper_parallel.trainer.config.replacement import ReplacementTarget
from hyper_parallel.trainer.config.resolver import ConfigResolutionError, replace_override_path, resolve_config
from tests.common.grouped_experts_family import TinyMoE, family_config, replace_experts
from tests.common.mark_utils import arg_mark
from tests.ut.components.quantization.test_mxfp8_memory import IdentityMXOps


class ExampleBaseline:
    """Test-only constructor retaining ordinary arguments and accepting **kwargs."""

    def __init__(self, width: int = 16, **kwargs: object) -> None:
        """Record arguments to detect selector leakage.

        Args:
            width: Ordinary constructor parameter shared by the implementations.
            **kwargs: Extra arguments that the strict replacement resolver must reject.
        """
        self.width = width
        self.extra = kwargs


class ExampleCompact(ExampleBaseline):
    """Test-only low-precision implementation without device operations."""


class ExampleFused(ExampleBaseline):
    """Test-only fused implementation without device operations."""


def _example_family() -> ModuleFamily:
    return ModuleFamily(
        name="example",
        baseline=ExampleBaseline,
        variants={"compact": f"{__name__}.ExampleCompact", "fused": f"{__name__}.ExampleFused"},
        selector_specs={"precision": SelectorSpec(str, "native"), "fusion": SelectorSpec(bool, False)},
        variant_rules=(
            VariantRule({"precision": "compact", "fusion": False}, "compact"),
            VariantRule({"fusion": True, "precision": "compact"}, "fused"),
        ),
    )


def _example_config(**arguments):
    raw = family_config().to_dict()
    raw["plan_overrides"][0]["replace_module"] = {"_target_": f"{__name__}.ExampleBaseline", **arguments}
    return resolve_config(yaml.safe_load(yaml.safe_dump(raw)))


class ModuleFamilySelectorTests(unittest.TestCase):
    """Keep registry mutations local; never mock selector or override resolution."""

    def setUp(self) -> None:
        """Restore exact registry contents after every test."""
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        registry = patch.dict(family_registry._FAMILIES)
        registry.start()
        self.addCleanup(registry.stop)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_grouped_selection_yaml_and_replacement(self) -> None:
        """
        Feature: Declarative ModuleFamily selector contracts.
        Description: Parse YAML and direct targets, then install the selected real expert modules.
        Expectation: Existing YAML combinations select the same class and install real modules.
        """
        scenarios = (
            ({}, GroupedExperts),
            ({"use_mxfp8": False}, GroupedExperts),
            ({"use_fused_mxfp8": False}, GroupedExperts),
            ({"use_mxfp8": False, "use_fused_mxfp8": False}, GroupedExperts),
            ({"use_mxfp8": True}, MXFP8GroupedSwiGLU),
            ({"use_mxfp8": True, "use_fused_mxfp8": False}, MXFP8GroupedSwiGLU),
            ({"use_mxfp8": True, "use_fused_mxfp8": True}, FusedMXFP8GroupedSwiGLU),
        )
        backend = "hyper_parallel.components.modules.mxfp8_grouped_swiglu"
        with patch(f"{backend}.validate_npu_gmm_runtime"), patch(f"{backend}.validate_npu_swiglu_runtime"), \
                patch.object(MXFP8Quantizer, "npu_ops", new_callable=PropertyMock, return_value=IdentityMXOps()):
            for selectors, expected in scenarios:
                with self.subTest(selectors=selectors):
                    config = resolve_config(yaml.safe_load(yaml.safe_dump(family_config(**selectors).to_dict())))
                    target = config.plan_overrides[0].replace_module
                    direct = ReplacementTarget(GroupedExperts,
                                               target_path="hyper_parallel.components.modules.GroupedExperts",
                                               **selectors)
                    self.assertIs(target.callable, expected)
                    self.assertIs(direct.callable, expected)
                    self.assertEqual(target.selectors, selectors)
                    self.assertEqual(direct.to_dict(), {"_target_": target.to_dict()["_target_"], **selectors})
                    serialized_selectors = {name: value for name, value in target.to_dict().items()
                                            if name in get_module_family(GroupedExperts).selector_specs}
                    self.assertEqual(serialized_selectors, selectors)
                    model = TinyMoE()
                    replace_experts(model, config)
                    self.assertIs(type(model.mlp.experts), expected)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_grouped_invalid_selectors(self) -> None:
        """
        Feature: Declarative ModuleFamily selector contracts.
        Description: Submit invalid combinations, non-boolean values and unknown constructor keys.
        Expectation: Strict bool types and rule-defined dependencies fail during configuration.
        """
        invalid = ({"use_fused_mxfp8": True}, {"use_mxfp8": False, "use_fused_mxfp8": True},
                   {"use_mxfp8": "true"}, {"use_mxfp8": 1}, {"use_mxfp8": 0},
                   {"use_fused_mxfp8": "false"}, {"typo": 1}, {"use_mxfp8": True, "typo": 1})
        for selectors in invalid:
            with self.subTest(selectors=selectors), self.assertRaises(ConfigResolutionError):
                family_config(**selectors)
        with self.assertRaisesRegex(ConfigResolutionError, "grouped_experts.*legal combinations"):
            family_config(use_fused_mxfp8=True)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_replace_preserves_explicit_fields_and_original(self) -> None:
        """
        Feature: Declarative ModuleFamily selector contracts.
        Description: Change both selectors through copies and serialize each configuration through YAML.
        Expectation: Effective defaults stay out of serialization and copies re-run selection.
        """
        original = family_config().plan_overrides[0].replace_module
        low = original.replace(use_mxfp8=True)
        fused = low.replace(use_fused_mxfp8=True)
        baseline = fused.replace(use_mxfp8=False, use_fused_mxfp8=False)
        self.assertIs(original.callable, GroupedExperts)
        self.assertEqual(original.selectors, {})
        self.assertEqual(low.selectors, {"use_mxfp8": True})
        self.assertIs(fused.callable, FusedMXFP8GroupedSwiGLU)
        self.assertIs(baseline.callable, GroupedExperts)
        self.assertEqual(baseline.selectors, {"use_mxfp8": False, "use_fused_mxfp8": False})
        for target in (original, low, fused, baseline):
            raw = family_config().to_dict()
            raw["plan_overrides"][0]["replace_module"] = target.to_dict()
            restored = resolve_config(yaml.safe_load(yaml.safe_dump(raw))).plan_overrides[0].replace_module
            self.assertIs(restored.callable, target.callable)
            self.assertEqual(restored.to_dict(), target.to_dict())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_custom_schema_configuration_and_target_overrides(self) -> None:
        """
        Feature: Declarative ModuleFamily selector contracts.
        Description: Register a string/bool schema and apply Target-level selector and ordinary argument changes.
        Expectation: Different selector names and types work through the same public entry points.
        """
        register_module_family(_example_family())
        config = _example_config(width=24)
        original = config.plan_overrides[0].replace_module
        low = replace_override_path(original, ["precision"], "compact", path="replace_module")
        fused = replace_override_path(low, ["fusion"], True, path="replace_module")
        resized = replace_override_path(fused, ["width"], 48, path="replace_module")
        for target, expected, width in ((original, ExampleBaseline, 24), (low, ExampleCompact, 24),
                                        (fused, ExampleFused, 24), (resized, ExampleFused, 48)):
            with self.subTest(expected=expected, width=width):
                instance = target.build()
                self.assertIs(type(instance), expected)
                self.assertEqual(instance.width, width)
                self.assertEqual(instance.extra, {})
                raw = config.to_dict()
                raw["plan_overrides"][0]["replace_module"] = target.to_dict()
                restored = resolve_config(yaml.safe_load(yaml.safe_dump(raw))).plan_overrides[0].replace_module
                self.assertIs(restored.callable, expected)
                self.assertEqual(restored.to_dict(), target.to_dict())
        self.assertEqual(original.selectors, {})
        explicit = _example_config(precision="native", fusion=False).plan_overrides[0].replace_module
        self.assertIs(explicit.callable, ExampleBaseline)
        self.assertEqual(explicit.selectors, {"precision": "native", "fusion": False})

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_custom_schema_rejects_unknown_combinations_types_and_keywords(self) -> None:
        """
        Feature: Declarative ModuleFamily selector contracts.
        Description: Exercise real parsing and Target overrides against the custom schema.
        Expectation: Constructors with **kwargs cannot absorb typos or invalid selectors.
        """
        register_module_family(_example_family())
        invalid = ({"precision": "missing"}, {"fusion": True}, {"precision": True},
                   {"precision": 1}, {"fusion": 0}, {"fusion": "true"}, {"unknown": 1},
                   {"precision": "compact", "unknown": 1})
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ConfigResolutionError):
                _example_config(**arguments)
        target = _example_config().plan_overrides[0].replace_module
        for name, value in (("precision", True), ("fusion", 1), ("fusion", "true")):
            with self.subTest(name=name, value=value), self.assertRaises(ConfigResolutionError):
                replace_override_path(target, [name], value, path="replace_module")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_order_independence_and_argument_immutability(self) -> None:
        """
        Feature: Declarative ModuleFamily selector contracts.
        Description: Reverse schema and argument insertion order while keeping names and values identical.
        Expectation: Matching compares complete named mappings, not tuple insertion order.
        """
        family = _example_family()
        register_module_family(replace(family, selector_specs=dict(reversed(list(family.selector_specs.items())))))
        for arguments in ({"width": 7, "fusion": True, "precision": "compact"},
                          {"precision": "compact", "fusion": True, "width": 7}):
            original = arguments.copy()
            target, kwargs = resolve_module_family_target(ExampleBaseline, arguments)
            self.assertIs(target, ExampleFused)
            self.assertEqual(kwargs, {"width": 7})
            self.assertEqual(arguments, original)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_registration_is_lazy_and_three_field_family_is_compatible(self) -> None:
        """
        Feature: Declarative ModuleFamily selector contracts.
        Description: Observe real import calls and register a legacy declaration without a schema.
        Expectation: Baseline selection and registration never import declared variant paths.
        """
        family = replace(_example_family(), variants={"compact": "not_installed.Low", "fused": "not_installed.Fused"})
        with patch.object(family_registry, "import_module", wraps=import_module) as importer:
            register_module_family(family)
            self.assertIs(resolve_module_family_target(ExampleBaseline, {})[0], ExampleBaseline)
            importer.assert_not_called()
            with self.assertRaises(ModuleNotFoundError):
                resolve_module_family_target(ExampleBaseline, {"precision": "compact"})
            importer.assert_called_once_with("not_installed")
        legacy = ModuleFamily("legacy", ExampleCompact, {"unused": "not_installed.Unused"})
        register_module_family(legacy)
        self.assertIs(resolve_module_family_target(ExampleCompact, {"width": 3})[0], ExampleCompact)
        self.assertEqual(legacy.selector_specs, {})

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_exact_baseline_identity_and_nonfamily_selectors(self) -> None:
        """
        Feature: Declarative ModuleFamily selector contracts.
        Description: Use registered selector names with subclasses and a distinct same-named class.
        Expectation: Subclasses and same-named classes do not inherit registered selectors.
        """
        register_module_family(_example_family())
        same_name = type("ExampleBaseline", (ExampleBaseline,), {})
        for target in (ExampleCompact, same_name):
            self.assertIsNone(get_module_family(target))
            for selector, value in (("precision", "native"), ("fusion", False), ("use_mxfp8", False)):
                with self.subTest(target=target, selector=selector), self.assertRaisesRegex(
                        ValueError, "registered ModuleFamily"):
                    resolve_module_family_target(target, {selector: value})
        with self.assertRaisesRegex(TypeError, "unknown replacement arguments"):
            resolve_module_family_target(ExampleCompact, {"unknown": 1})

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_invalid_registration_schemas(self) -> None:
        """
        Feature: Declarative ModuleFamily selector contracts.
        Description: Register non-callable baselines, invalid names, invalid types and invalid defaults.
        Expectation: Type declarations, selector names and defaults are checked before insertion.
        """
        family = _example_family()
        invalid = (
            replace(family, baseline=42),
            replace(family, selector_specs={"": SelectorSpec(bool, False)}),
            replace(family, selector_specs={1: SelectorSpec(bool, False)}),
            replace(family, selector_specs={"precision": SelectorSpec("str", "native")}),
            replace(family, selector_specs={"fusion": SelectorSpec(bool, 0)}),
            replace(family, selector_specs={"fusion": SelectorSpec(bool, "false")}),
            replace(family, selector_specs={"precision": SelectorSpec(str, False)}),
        )
        for declaration in invalid:
            with self.subTest(declaration=declaration), self.assertRaises((TypeError, ValueError)):
                register_module_family(declaration)
            self.assertIsNone(get_module_family(ExampleBaseline))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_invalid_rules_and_duplicate_registration(self) -> None:
        """
        Feature: Declarative ModuleFamily selector contracts.
        Description: Register incomplete, mistyped, default and duplicate conditions or undeclared variants.
        Expectation: Rules require complete typed keys, declared variants and unique nondefaults.
        """
        family = _example_family()
        invalid_rules = (
            (VariantRule({"precision": "compact"}, "compact"),),
            (VariantRule({"precision": "compact", "fusion": False, "extra": 1}, "compact"),),
            (VariantRule({"precision": "compact", "fusion": 0}, "compact"),),
            (VariantRule({"precision": 1, "fusion": False}, "compact"),),
            (VariantRule({"precision": "compact", "fusion": False}, "missing"),),
            (VariantRule({"precision": "native", "fusion": False}, "compact"),),
            (*family.variant_rules, VariantRule({"fusion": False, "precision": "compact"}, "fused")),
        )
        for rules in invalid_rules:
            with self.subTest(rules=rules), self.assertRaises((TypeError, ValueError)):
                register_module_family(replace(family, variant_rules=rules))
            self.assertIsNone(get_module_family(ExampleBaseline))
        register_module_family(family)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_module_family(family)

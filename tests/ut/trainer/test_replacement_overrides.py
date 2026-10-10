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
"""End-to-end CLI contracts for Module Family replacement selections."""

import copy
import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from hyper_parallel.components.modules import GroupedExperts, MXFP8GroupedSwiGLU
from hyper_parallel.components.modules.mxfp8_grouped_swiglu import FusedMXFP8GroupedSwiGLU
from hyper_parallel.models.replacement import _apply_module_replacement_actions
from hyper_parallel.trainer.config import entries_to_module_replacements
from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.trainer.config.resolver import ConfigResolutionError
from tests.common.grouped_experts_family import TinyMoE


_TARGET_PATH = "hyper_parallel.components.modules.GroupedExperts"
_SELECTOR_PREFIX = "--plan_overrides.0.replace_module."


def _training_config(**selectors: bool) -> dict:
    """Return a full Trainer input rather than an isolated Target fixture."""
    return {
        "model": {"_target_": "tests.common.grouped_experts_family.TinyMoE"},
        "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.01},
        "training": {"train_iters": 3, "backend": "gloo", "init_device": "cpu"},
        "accelerator": {"tp_size": 1, "cp_size": 1, "ep_size": 1, "pp_size": 1},
        "plan_overrides": [{
            "match": "mlp.experts",
            "module_type": "tests.common.grouped_experts_family.PackedExperts",
            "replace_module": {"_target_": _TARGET_PATH, **selectors},
        }],
    }


class ReplacementOverrideTests(unittest.TestCase):
    """Exercise the production parser and replacement without device arithmetic."""

    def setUp(self) -> None:
        """Keep RNG state and temporary files isolated between tests."""
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())

    def _parse(self, raw: dict, *overrides: str):
        """Load a real YAML file through the public training entry point."""
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "trainer.yaml"
            config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            return parse_training_args([str(config_path), *overrides])

    def test_all_variants_reach_real_replacement(self) -> None:
        """CLI-selected implementations preserve loaded weights on execution."""
        cases = (
            ((), GroupedExperts),
            (("use_mxfp8=true",), MXFP8GroupedSwiGLU),
            (("use_fused_mxfp8=true", "use_mxfp8=true"), FusedMXFP8GroupedSwiGLU),
        )
        runtime = "hyper_parallel.components.modules.mxfp8_grouped_swiglu"
        with patch(f"{runtime}.validate_npu_gmm_runtime"), patch(f"{runtime}.validate_npu_swiglu_runtime"):
            for selectors, expected in cases:
                with self.subTest(variant=expected.__name__):
                    config = self._parse(_training_config(), *(_SELECTOR_PREFIX + item for item in selectors))
                    model = TinyMoE()
                    original = copy.deepcopy(model.mlp.experts.state_dict())
                    _apply_module_replacement_actions(
                        model, entries_to_module_replacements(config.plan_overrides), weights_mapping=[]
                    )
                    self.assertIs(type(model.mlp.experts), expected)
                    torch.testing.assert_close(
                        model.mlp.experts.gate_up_proj, original["gate_up_proj"].transpose(-1, -2)
                    )
                    torch.testing.assert_close(model.mlp.experts.down_proj, original["down_proj"].transpose(-1, -2))

    def test_selector_batch_is_order_independent(self) -> None:
        """Enabling and disabling both selectors validate the final pair."""
        for enabled, expected in ((True, FusedMXFP8GroupedSwiGLU), (False, GroupedExperts)):
            text = str(enabled).lower()
            flags = (f"use_mxfp8={text}", f"use_fused_mxfp8={text}")
            initial = _training_config(use_mxfp8=not enabled, use_fused_mxfp8=not enabled)
            for order in itertools.permutations(flags):
                with self.subTest(enabled=enabled, order=order):
                    config = self._parse(initial, *(_SELECTOR_PREFIX + flag for flag in order))
                    self.assertIs(config.plan_overrides[0].replace_module.callable, expected)

    def test_round_trip_and_reselection(self) -> None:
        """Serialized configs retain the baseline target and consumed selectors."""
        for low, fused in ((False, False), (True, False), (True, True)):
            with self.subTest(low=low, fused=fused):
                config = self._parse(_training_config(), _SELECTOR_PREFIX + f"use_mxfp8={str(low).lower()}",
                                     _SELECTOR_PREFIX + f"use_fused_mxfp8={str(fused).lower()}")
                serialized = config.to_dict()
                target = serialized["plan_overrides"][0]["replace_module"]
                self.assertEqual(target["_target_"], _TARGET_PATH)
                self.assertEqual((target["use_mxfp8"], target["use_fused_mxfp8"]), (low, fused))
                restored = self._parse(serialized)
                self.assertIs(restored.plan_overrides[0].replace_module.callable,
                              config.plan_overrides[0].replace_module.callable)
                baseline = self._parse(serialized, _SELECTOR_PREFIX + "use_mxfp8=false",
                                       _SELECTOR_PREFIX + "use_fused_mxfp8=false")
                self.assertIs(baseline.plan_overrides[0].replace_module.callable, GroupedExperts)

    def test_invalid_final_pair_reports_cli_location(self) -> None:
        """A final fused-without-MXFP8 request cannot pass through intermediate state."""
        with self.assertRaisesRegex(ConfigResolutionError, r"CLI\.plan_overrides\.0\.replace_module:.*requires"):
            self._parse(_training_config(), _SELECTOR_PREFIX + "use_mxfp8=false",
                        _SELECTOR_PREFIX + "use_fused_mxfp8=true")

    def test_selectors_require_boolean_values(self) -> None:
        """Strings, numbers, nulls and collections cannot silently enable a selector."""
        for name in ("use_mxfp8", "use_fused_mxfp8"):
            for value in ("1", "0", "'true'", "null", "[]"):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ConfigResolutionError, rf"replace_module\.{name}: expected bool"):
                        self._parse(_training_config(), _SELECTOR_PREFIX + f"{name}={value}")

    def test_list_indices_are_explicit_and_bounded(self) -> None:
        """Invalid indices never grow, append to or wrap around the override list."""
        for index in ("-1", "1", "9", "first", "01"):
            with self.subTest(index=index):
                with self.assertRaisesRegex(ConfigResolutionError, "list index"):
                    self._parse(_training_config(), f"--plan_overrides.{index}.replace_module.use_mxfp8=true")
        empty = _training_config()
        empty["plan_overrides"] = []
        with self.assertRaisesRegex(ConfigResolutionError, "out of range for length 0"):
            self._parse(empty, _SELECTOR_PREFIX + "use_mxfp8=true")

    def test_missing_component_is_not_created(self) -> None:
        """A selector cannot create an undeclared replacement target implicitly."""
        raw = _training_config()
        del raw["plan_overrides"][0]["replace_module"]
        with self.assertRaisesRegex(ConfigResolutionError, "component was not selected"):
            self._parse(raw, _SELECTOR_PREFIX + "use_mxfp8=true")

    def test_unknown_fields_and_constructor_arguments_fail(self) -> None:
        """List traversal preserves strict field and replacement keyword validation."""
        for flag, message in (
            ("--plan_overrides.0.mtach=mlp.experts", "unknown field 'mtach'"),
            (_SELECTOR_PREFIX + "use_mxf8=true", "unknown replacement arguments"),
            (_SELECTOR_PREFIX + "_target_=other.callable", "changing _target_"),
        ):
            with self.subTest(flag=flag):
                with self.assertRaisesRegex(ConfigResolutionError, message):
                    self._parse(_training_config(), flag)

    def test_indices_select_independent_entries(self) -> None:
        """A second list entry and an ordinary typed field can change in one batch."""
        raw = _training_config()
        raw["plan_overrides"].append(copy.deepcopy(raw["plan_overrides"][0]))
        config = self._parse(raw, "--plan_overrides.1.replace_module.use_mxfp8=true",
                             "--plan_overrides.1.match=other.experts", "--accelerator.ep_size=2")
        self.assertIs(config.plan_overrides[0].replace_module.callable, GroupedExperts)
        self.assertIs(config.plan_overrides[1].replace_module.callable, MXFP8GroupedSwiGLU)
        self.assertEqual(config.plan_overrides[1].match, "other.experts")
        self.assertEqual(config.accelerator.ep_size, 2)

    def test_indexed_values_retain_item_types(self) -> None:
        """An index override cannot bypass the annotation on its parent list."""
        raw = _training_config()
        raw["plan_overrides"][0]["match"] = ["mlp.experts"]
        valid = self._parse(raw, "--plan_overrides.0.match.0=other.experts")
        self.assertEqual(valid.plan_overrides[0].match, ["other.experts"])
        with self.assertRaisesRegex(ConfigResolutionError, "expected"):
            self._parse(raw, "--plan_overrides.0.match.0=17")

    def test_repeated_selector_uses_last_value(self) -> None:
        """Repeated options retain normal command-line last-write semantics."""
        config = self._parse(_training_config(), _SELECTOR_PREFIX + "use_fused_mxfp8=true",
                             _SELECTOR_PREFIX + "use_fused_mxfp8=false", _SELECTOR_PREFIX + "use_mxfp8=true")
        self.assertIs(config.plan_overrides[0].replace_module.callable, MXFP8GroupedSwiGLU)


if __name__ == "__main__":
    unittest.main()

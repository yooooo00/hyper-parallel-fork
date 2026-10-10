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
"""Real planner/default adapter contracts without device arithmetic."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from hyper_parallel.components.modules.family import (
    get_module_family, get_module_family_selection, resolve_module_family_target,
)
from hyper_parallel.components.modules.gqa_attention import GQAAttention
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed._builder.module_family import validate_module_family_plan
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed._builder.forward_rewriter import _wrap_inner_attention
from hyper_parallel.distributed.context_parallel.attention_interface import (
    attention_interface_cp_wrapper, _cp_mask,
)
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.trainer.config import PlanOverride, entries_to_plan_overrides
from hyper_parallel.trainer.config.resolver import resolve_config
from tests.common.gqa_family import native_attention, replace_attention


class FamilyDefaultsTests(unittest.TestCase):
    """Keep configuration, selection, planner merge and forward rewrite real."""

    def _model(self, kind="llama"):
        model = nn.Module()
        model.self_attn = replace_attention(native_attention(kind))[0]
        model.config = model.self_attn.config
        return model

    @staticmethod
    def _mesh():
        return init_device_mesh("cpu", (2,), mesh_dim_names=("cp",), rank_list=(0, 1), init_backend=False)

    def test_default_and_explicit_priority(self):
        """Two native models share one default; explicit wrappers win over auto."""
        mesh = self._mesh()
        for kind in ("llama", "qwen3_moe"):
            model = self._model(kind)
            for value in (None, "auto", attention_interface_cp_wrapper):
                override = {"self_attn": ModuleShardingSpec(inner_wrapper=value)}
                plan = ShardingPlanner(plan_overrides=override).plan(model, mesh, cp_size=2)
                self.assertIs(plan.modules["self_attn"].inner_wrapper, attention_interface_cp_wrapper)
                expected = "explicit" if callable(value) else "family default"
                self.assertEqual(plan.family_modules["self_attn"]["adapter_source"], expected)
                self.assertTrue(plan.family_modules["self_attn"]["tp_replicated_gradient_contract"])
                self.assertIn("gqa_attention", plan.explain())
                self.assertEqual(get_module_family_selection(model.self_attn).variant, "baseline")

    def test_explicit_disable_is_not_overwritten(self):
        """Both public sentinels survive parsing and trigger a useful capability error."""
        for value in (False, "none"):
            config = resolve_config({"model": {"_target_": "torch.nn.Identity"},
                                     "optimizer": {"_target_": "torch.optim.SGD", "lr": .1},
                                     "plan_overrides": [{"match": "self_attn", "inner_wrapper": value}]})
            overrides = entries_to_plan_overrides(config.plan_overrides, cp_size=2)
            with self.assertRaisesRegex(ValueError, "self_attn.*gqa_attention.*explicitly disabled"):
                ShardingPlanner(plan_overrides=overrides).plan(self._model(), self._mesh(), cp_size=2)

    def test_inserted_boundary_family_defaults_and_summary(self) -> None:
        """A complete derive=False contract may inherit or explicitly select its Family adapter."""
        model, mesh = self._model(), self._mesh()
        explicit = {"_target_": (
            "hyper_parallel.distributed.context_parallel.attention_interface.attention_interface_cp_wrapper"
        )}
        for wrapper in (None, "auto", explicit):
            with self.subTest(wrapper=wrapper):
                config = self._explicit_cp_config(model, wrapper)
                overrides = entries_to_plan_overrides(config.plan_overrides, cp_size=2)
                planner = ShardingPlanner(plan_overrides=overrides, derive=False)
                for _ in range(2):
                    plan = planner.plan(model, mesh, cp_size=2)
                    self.assertEqual(set(plan.modules), {"self_attn"})
                    spec = plan.modules["self_attn"]
                    selected = getattr(spec.inner_wrapper, "callable", spec.inner_wrapper)
                    self.assertIs(selected, attention_interface_cp_wrapper)
                    self.assertEqual(spec.inner_target, "self")
                    self.assertFalse(spec.region_dispatch)
                    summary = plan.family_modules["self_attn"]
                    expected_source = "explicit" if isinstance(wrapper, dict) else "family default"
                    self.assertEqual(summary["adapter_source"], expected_source)
                    self.assertEqual(summary["parallel_sizes"], {"tp": 1, "cp": 2, "ep": 1})
                    self.assertTrue(summary["tp_replicated_gradient_contract"])
                    self.assertIn("gqa_attention", plan.explain())
                original_wrapper = overrides["self_attn"].inner_wrapper
                if isinstance(wrapper, dict):
                    self.assertEqual(original_wrapper.to_dict(), wrapper)
                else:
                    self.assertEqual(original_wrapper, wrapper)

    def test_inserted_boundary_explicit_disable_is_rejected(self) -> None:
        """An inserted none/False adapter remains disabled through Family defaults."""
        model, mesh = self._model(), self._mesh()
        for wrapper in (False, "none"):
            with self.subTest(wrapper=wrapper):
                config = self._explicit_cp_config(model, wrapper)
                overrides = entries_to_plan_overrides(config.plan_overrides, cp_size=2)
                with self.assertRaisesRegex(ValueError, "self_attn.*gqa_attention.*explicitly disabled"):
                    ShardingPlanner(plan_overrides=overrides, derive=False).plan(model, mesh, cp_size=2)
                self.assertEqual(overrides["self_attn"].inner_wrapper, wrapper)

    @staticmethod
    def _explicit_cp_config(model: nn.Module, wrapper):
        """Parse a fully declared CP boundary through the public Trainer config schema."""
        return resolve_config({
            "model": {"_target_": "torch.nn.Identity"},
            "optimizer": {"_target_": "torch.optim.SGD", "lr": .1},
            "plan_overrides": [{
                "match": "self_attn",
                "params": {name: {"cp": "replicate"} for name, _ in model.self_attn.named_parameters()},
                "in_src": {"hidden_states": {"cp": "shard(1)"}},
                "in_dst": {"hidden_states": {"cp": "shard(1)"}},
                "out_src": {"cp": "shard(1)"}, "out_dst": {"cp": "shard(1)"},
                "inner_wrapper": wrapper,
            }],
        })

    def test_missing_variant_and_round_trip(self):
        """No silent fused GQA fallback and no bare missing-variant dictionary error."""
        source = native_attention("llama")
        with self.assertRaisesRegex(ValueError, "gqa_attention.*use_fused_mxfp8.*unavailable"):
            replace_attention(source, use_mxfp8=True, use_fused_mxfp8=True)
        config = resolve_config({"model": {"_target_": "torch.nn.Identity"},
                                 "optimizer": {"_target_": "torch.optim.SGD", "lr": .1},
                                 "plan_overrides": [PlanOverride(match="self_attn", inner_wrapper="auto")]})
        self.assertEqual(resolve_config(config.to_dict()).to_dict(), config.to_dict())

    def test_adapter_install_once_and_contract_guards(self):
        """Use the production rewriter; a second install cannot wrap the saved interface."""
        model = self._model()
        mesh = self._mesh()
        plan = ShardingPlanner().plan(model, mesh, cp_size=2)
        spec = plan.modules["self_attn"]
        original = model.self_attn.attention_interface
        for _ in range(2):
            _wrap_inner_attention(model.self_attn, mesh, spec=spec, mesh=mesh,
                                  mesh_dim_names=("cp",), module_fqn="self_attn")
            self.assertIs(model.self_attn._hp_attention_cp_original_interface, original)
        with self.assertRaisesRegex(ValueError, "cache"):
            model.self_attn(torch.empty(1, 4, 128), past_key_values=object())
        with self.assertRaisesRegex(ValueError, "packed"):
            model.self_attn(torch.empty(1, 4, 128), actual_seq_len=[4])

    def test_global_masks_and_causal_offsets(self):
        """Padding/custom masks retain global coordinates on the second CP rank."""
        query = torch.empty(2, 4, 3, 32)
        causal = torch.arange(6)[None, :] <= torch.arange(3, 6)[:, None]
        torch.testing.assert_close(_cp_mask(None, query, 6, 3, True), causal)
        padding = torch.tensor([[True] * 5 + [False], [True] * 6])
        expected = causal & padding[:, None, None, :]
        torch.testing.assert_close(_cp_mask(padding, query, 6, 3, True), expected)
        custom = torch.ones(2, 1, 6, 6, dtype=torch.bool)
        custom[..., 2] = False
        torch.testing.assert_close(_cp_mask(custom, query, 6, 3, True), causal & custom[..., 3:, :])
        with self.assertRaisesRegex(ValueError, "global key"):
            _cp_mask(torch.ones(2, 3, dtype=torch.bool), query, 6, 3, True)
        with self.assertRaisesRegex(ValueError, "global query"):
            _cp_mask(torch.ones(2, 1, 3, 6, dtype=torch.bool), query, 6, 3, True)

    def test_invalid_mask_and_hidden_state_dimensions(self) -> None:
        """Malformed tensors fail clearly before indexing shapes or communicating."""
        query = torch.empty(2, 4, 3, 32)
        for shape in ((), (6,), (2, 3, 6), (1, 2, 1, 3, 6)):
            with self.subTest(mask_shape=shape):
                with self.assertRaisesRegex(ValueError, "attention_mask.*2 or 4 dimensions"):
                    _cp_mask(torch.zeros(shape), query, 6, 3, True)

        model, mesh = self._model(), self._mesh()
        spec = ShardingPlanner().plan(model, mesh, cp_size=2).modules["self_attn"]
        _wrap_inner_attention(model.self_attn, mesh, spec=spec, mesh=mesh,
                              mesh_dim_names=("cp",), module_fqn="self_attn")
        with patch("hyper_parallel.distributed.context_parallel.attention_interface.dist.all_gather_object") as gather:
            for shape in ((), (128,), (4, 128), (1, 1, 4, 128)):
                with self.subTest(hidden_shape=shape):
                    with self.assertRaisesRegex(ValueError, "hidden_states.*3 dimensions"):
                        model.self_attn(torch.empty(shape))
            gather.assert_not_called()

    def test_all_packed_length_aliases_fail_before_communication(self) -> None:
        """Nested metadata and every supported length alias cannot silently enable packed CP."""
        aliases = (
            "packed_seq_params", "actual_seq_len", "actual_q_len", "actual_kv_len",
            "actual_seq_qlen", "actual_seq_kvlen", "cu_seq_lens", "cu_seq_lens_q",
            "cu_seq_lens_k", "cu_seq_lens_kv", "cu_seqlens_q", "cu_seqlens_k", "cu_seqlens_kv",
        )
        model, mesh = self._model(), self._mesh()
        spec = ShardingPlanner().plan(model, mesh, cp_size=2).modules["self_attn"]
        _wrap_inner_attention(model.self_attn, mesh, spec=spec, mesh=mesh,
                              mesh_dim_names=("cp",), module_fqn="self_attn")
        hidden = torch.empty(1, 4, 128)
        states = torch.empty(1, 4, 4, 32)
        with patch("hyper_parallel.distributed.context_parallel.attention_interface.dist.all_gather_object") as gather:
            for alias in aliases:
                with self.subTest(alias=alias):
                    metadata = {alias: [0, 4]}
                    with self.assertRaisesRegex(ValueError, f"packed sequence metadata.*{alias}"):
                        model.self_attn(hidden, **metadata)
                    with self.assertRaisesRegex(ValueError, f"packed sequence metadata.*{alias}"):
                        model.self_attn.attention_interface(model.self_attn, states, states, states, None, **metadata)
            gather.assert_not_called()

    def test_exact_identity_and_unknown_family(self) -> None:
        """Neither subclasses nor matching type-name strings extend registration."""
        derived = type("DerivedGQA", (GQAAttention,), {})
        impostor = type(GQAAttention.__name__, (nn.Module,), {"__module__": GQAAttention.__module__})
        for target in (derived, impostor, nn.Linear):
            with self.subTest(target=target):
                self.assertIsNone(get_module_family(target))
                with self.assertRaisesRegex(ValueError, "registered ModuleFamily"):
                    resolve_module_family_target(target, {"use_mxfp8": True})

    @staticmethod
    def _low_model(compact: bool = False) -> nn.Module:
        source = native_attention("llama")
        if compact:
            source.config.hidden_size = 96
            source.config.num_attention_heads = 6
            source.config.head_dim = 16
            source = type(source)(source.config, layer_idx=0).to(dtype=torch.bfloat16)
        with patch("hyper_parallel.components.modules.mxfp8_gqa_attention.validate_npu_runtime"):
            attention, _ = replace_attention(source, use_mxfp8=True)
        model = nn.Module()
        model.self_attn = attention
        model.config = attention.config
        return model

    @staticmethod
    def _tp_mesh():
        return init_device_mesh("cpu", (2,), mesh_dim_names=("tp",), rank_list=(0, 1), init_backend=False)

    def test_gqa_final_parameter_boundaries(self) -> None:
        """Every final full parameter path is checked, including child overrides."""
        model, mesh = self._low_model(), self._tp_mesh()
        plan = ShardingPlanner().plan(model, mesh, tp_size=2)
        placements = plan.modules["self_attn"].params.pop("linear_qkv.weight")
        child = ModuleShardingSpec(params={"weight": placements})
        plan.modules["self_attn.linear_qkv"] = child
        validate_module_family_plan(model, plan, mesh, None, validate_mode=False)
        child.params["weight"] = {"tp": Shard(1)}
        with self.assertRaisesRegex(ValueError, "self_attn.*linear_qkv.*unsupported projection placement"):
            validate_module_family_plan(model, plan, mesh, None, validate_mode=False)
        child.params["weight"] = placements
        plan.modules["self_attn"].params["linear_qkv.weight"] = placements
        with self.assertRaisesRegex(ValueError, "self_attn.linear_qkv.weight exactly once"):
            validate_module_family_plan(model, plan, mesh, None, validate_mode=False)

    def test_gqa_local_alignment_and_required_tp_roles(self) -> None:
        """Globally aligned weights must also align after TP; roles cannot disappear."""
        model, mesh = self._low_model(compact=True), self._tp_mesh()
        plan = ShardingPlanner().plan(model, mesh, tp_size=2)
        with self.assertRaisesRegex(ValueError, "self_attn.*local linear_qkv.weight.*aligned to 32"):
            validate_module_family_plan(model, plan, mesh, None, validate_mode=False)

        model = self._low_model()
        plan = ShardingPlanner().plan(model, mesh, tp_size=2)
        plan.modules["self_attn"].params["linear_qkv.weight"] = {"tp": Replicate()}
        with self.assertRaisesRegex(ValueError, "self_attn.*requires TP Shard"):
            validate_module_family_plan(model, plan, mesh, None, validate_mode=False)
        del plan.modules["self_attn"].params["linear_qkv.weight"]
        with self.assertRaisesRegex(ValueError, "self_attn.linear_qkv.weight exactly once"):
            validate_module_family_plan(model, plan, mesh, None, validate_mode=False)

    def test_baseline_tp_validate_dispatch_guard(self) -> None:
        """The real mixed RoPE dispatch gap fails during build with its module FQN."""
        for kind in ("llama", "qwen3_moe"):
            model, mesh = self._model(kind), self._tp_mesh()
            plan = ShardingPlanner().plan(model, mesh, tp_size=2)
            validate_module_family_plan(model, plan, mesh, None, validate_mode=False)
            with self.assertRaisesRegex(NotImplementedError, "self_attn.*RoPE DTensor dispatch"):
                validate_module_family_plan(model, plan, mesh, None, validate_mode=True)

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
"""CPU contracts exercised through configuration, replacement and EP factories."""

import copy
import os
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import PropertyMock, patch

import torch
from torch import nn

from hyper_parallel.components.modules import GroupedExperts, MXFP8GroupedSwiGLU
from hyper_parallel.components.quantization.modules import MXFP8GroupedExperts
from hyper_parallel.components.modules.mxfp8_grouped_swiglu import FusedMXFP8GroupedSwiGLU
from hyper_parallel.components.quantization.functional.npu_mxfp8 import MXFP8NpuOps
from hyper_parallel.components.quantization.quantizers.mxfp8 import MXFP8Quantizer
from hyper_parallel.models.replacement import _apply_module_replacement_actions, module_replacement
from hyper_parallel.distributed._builder.module_family import validate_module_family_plan
from hyper_parallel.distributed.plan import ShardingPlan
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec, EP
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.trainer.config import PlanOverride, Target, entries_to_module_replacements
from hyper_parallel.trainer.config.resolver import ConfigResolutionError, replace_override_path, resolve_config
from tests.common.grouped_experts_family import TinyMoE, family_config, replace_experts
from tests.common.mark_utils import arg_mark
from tests.ut.components.quantization.test_mxfp8_memory import IdentityMXOps


class FakeEpMesh:
    """Isolate process-group lookup; selection and binding remain real."""

    def __getitem__(self, key: str) -> "FakeEpMesh":
        """Return metadata for the EP dimension."""
        del key
        return self

    def get_group(self, key: str) -> None:
        """Avoid creating real process groups in CPU contract tests."""
        del key

    def size(self) -> int:
        """Provide a two-rank topology for the binder."""
        return 2


class ModuleFamilyTests(unittest.TestCase):
    """Do not substitute the family resolver, replacement executor or EP binder."""

    def setUp(self) -> None:
        """Isolate only device arithmetic, runtime capability checks and RNG state."""
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(793)
        runtime = patch("hyper_parallel.components.modules.mxfp8_grouped_swiglu.validate_npu_gmm_runtime")
        runtime.start()
        self.addCleanup(runtime.stop)
        ops = patch.object(MXFP8Quantizer, "npu_ops", new_callable=PropertyMock, return_value=IdentityMXOps())
        ops.start()
        self.addCleanup(ops.stop)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_fused_selection_and_capability(self):
        """
        Feature: Fused GroupedExperts family selection.
        Description: Select fusion through real configuration and replacement, isolating only device capability.
        Expectation: Canonical weights and inherited routing remain intact; missing kernels report the module FQN.
        """
        config = family_config(use_mxfp8=True, use_fused_mxfp8=True)
        with patch("hyper_parallel.components.modules.mxfp8_grouped_swiglu.validate_npu_swiglu_runtime"):
            model = TinyMoE()
            replace_experts(model, config)
            self.assertIs(type(model.mlp.experts), FusedMXFP8GroupedSwiGLU)
            self.assertEqual(set(model.mlp.experts.state_dict()), {"gate_up_proj", "down_proj"})
            self.assertIs(FusedMXFP8GroupedSwiGLU.forward, GroupedExperts.forward)
            self.assertIs(FusedMXFP8GroupedSwiGLU.make_transforms, GroupedExperts.make_transforms)
        with patch("hyper_parallel.components.modules.mxfp8_grouped_swiglu.validate_npu_swiglu_runtime",
                          side_effect=RuntimeError("missing fused kernels")):
            with self.assertRaisesRegex(RuntimeError, "mlp.experts.*missing fused kernels"):
                replace_experts(TinyMoE(), config)
        fake = SimpleNamespace(_torch_npu=SimpleNamespace())
        with self.assertRaisesRegex(RuntimeError, "requires"):
            MXFP8NpuOps.validate_swiglu_fusion(fake)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_selection_and_conversion(self):
        """
        Feature: GroupedExperts module family.
        Description: Baseline and MXFP8 share persistent names, transforms, dtype and training state.
        Expectation: Accepted contracts hold and unsupported requests fail explicitly.
        """
        for canonical in (False, True):
            for low in (False, True):
                with self.subTest(canonical=canonical, low=low):
                    model = TinyMoE(canonical=canonical).eval()
                    source = model.mlp.experts
                    source.down_proj.requires_grad_(False)
                    original = copy.deepcopy(source.state_dict())
                    mapping = replace_experts(model, family_config(use_mxfp8=low, use_fused_mxfp8=False))
                    experts = model.mlp.experts
                    self.assertIs(type(experts), MXFP8GroupedSwiGLU if low else GroupedExperts)
                    self.assertEqual(set(experts.state_dict()), {"gate_up_proj", "down_proj"})
                    self.assertEqual(experts.gate_up_proj.shape, (4, 128, 512))
                    self.assertEqual(experts.down_proj.shape, (4, 256, 128))
                    self.assertFalse(experts.training)
                    self.assertFalse(experts.down_proj.requires_grad)
                    self.assertEqual(len(mapping), 0 if canonical else 2)
                    for name, value in experts.state_dict().items():
                        expected = original[name] if canonical else original[name].transpose(-2, -1)
                        torch.testing.assert_close(value, expected, rtol=0, atol=0)
                    for transform in mapping:
                        name = transform.source_patterns[0]
                        reverse = transform.operations[0].reverse_op
                        restored = reverse.convert({name: [experts.state_dict()[name]]},
                                                   source_patterns=[name], target_patterns=[name])[name]
                        torch.testing.assert_close(restored, original[name], rtol=0, atol=0)
                    if canonical:
                        self.assertIs(source.gate_up_proj, experts.gate_up_proj)
        self.assertIs(MXFP8GroupedSwiGLU.forward, GroupedExperts.forward)
        self.assertIs(MXFP8GroupedSwiGLU.forward_expert_major, GroupedExperts.forward_expert_major)
        self.assertIs(MXFP8GroupedSwiGLU.make_transforms, GroupedExperts.make_transforms)
        self.assertFalse(issubclass(MXFP8GroupedExperts, GroupedExperts))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_loaded_weights_survive_replacement_without_reload(self):
        """
        Feature: Replacement after single-device native checkpoint loading.
        Description: Execute the production replacement entry without the test helper's checkpoint conversion.
        Expectation: Baseline and both MXFP8 variants preserve already loaded source values and freezing.
        """
        for low, fused in ((False, False), (True, False), (True, True)):
            with self.subTest(low=low, fused=fused):
                model = TinyMoE().eval()
                source = model.mlp.experts
                source.down_proj.requires_grad_(False)
                expected = {name: value.detach().transpose(-2, -1).clone()
                            for name, value in source.named_parameters()}
                config = family_config(use_mxfp8=low, use_fused_mxfp8=fused)
                _apply_module_replacement_actions(
                    model, entries_to_module_replacements(config.plan_overrides), weights_mapping=[])
                for name, value in model.mlp.experts.named_parameters():
                    torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
                self.assertFalse(model.mlp.experts.down_proj.requires_grad)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_loaded_baseline_bias_survives_renaming(self):
        """
        Feature: Loaded baseline parameters with source bias names.
        Description: Replace a materialized baseline through the production executor.
        Expectation: Bias values and frozen state survive renaming without a checkpoint reload.
        """
        model = TinyMoE()
        source = model.mlp.experts
        source.gate_up_proj_bias = nn.Parameter(torch.randn(4, 512), requires_grad=False)
        source.down_proj_bias = nn.Parameter(torch.randn(4, 128))
        config = family_config()
        _apply_module_replacement_actions(
            model, entries_to_module_replacements(config.plan_overrides), weights_mapping=[])
        torch.testing.assert_close(model.mlp.experts.bias1, source.gate_up_proj_bias, rtol=0, atol=0)
        torch.testing.assert_close(model.mlp.experts.bias2, source.down_proj_bias, rtol=0, atol=0)
        self.assertFalse(model.mlp.experts.bias1.requires_grad)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_overrides_and_python_target(self):
        """
        Feature: GroupedExperts module family.
        Description: Selectors survive serialization and CLI/Python changes without leaking to baseline.
        Expectation: Accepted contracts hold and unsupported requests fail explicitly.
        """
        config = family_config()
        target = config.plan_overrides[0].replace_module
        self.assertIs(target.callable, GroupedExperts)
        low = replace_override_path(target, ["use_mxfp8"], True, path="replace_module")
        self.assertIs(low.callable, MXFP8GroupedSwiGLU)
        self.assertIs(low.replace(use_mxfp8=False).callable, GroupedExperts)
        self.assertIs(target.callable, GroupedExperts)
        raw = config.to_dict()
        raw["plan_overrides"][0]["replace_module"] = low.to_dict()
        self.assertIs(resolve_config(raw).plan_overrides[0].replace_module.callable, MXFP8GroupedSwiGLU)
        config.plan_overrides[0].replace_module = Target(
            GroupedExperts, target_path="hyper_parallel.components.modules.GroupedExperts", use_mxfp8=True)
        model = TinyMoE()
        replace_experts(model, config)
        self.assertIsInstance(model.mlp.experts, MXFP8GroupedSwiGLU)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_invalid_configuration(self):
        """
        Feature: GroupedExperts module family.
        Description: Fusion, non-boolean selectors and unknown arguments fail in the real parser.
        Expectation: Accepted contracts hold and unsupported requests fail explicitly.
        """
        for kwargs in ({"use_fused_mxfp8": True}, {"use_mxfp8": "true"}, {"typo": 1}, {"use_mxfp8": True, "typo": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ConfigResolutionError):
                family_config(**kwargs)
        raw = family_config().to_dict()
        raw["plan_overrides"][0]["replace_module"] = {"_target_": "torch.nn.Linear", "use_mxfp8": False}
        with self.assertRaisesRegex(ConfigResolutionError, "registered ModuleFamily"):
            resolve_config(raw)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_unsupported_structures_and_parallel_axes(self):
        """
        Feature: GroupedExperts module family.
        Description: Unsupported source or topology errors name the matched module FQN.
        Expectation: Accepted contracts hold and unsupported requests fail explicitly.
        """
        for field, value in (("router_gating_in_fp32", False), ("is_concatenated", False), ("has_bias", True)):
            model = TinyMoE()
            setattr(model.mlp.experts, field, value)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "mlp.experts"):
                replace_experts(model, family_config(use_mxfp8=True))
        for axis in ("pp",):
            with self.subTest(axis=axis), self.assertRaisesRegex(NotImplementedError, "mlp.experts"):
                replace_experts(TinyMoE(), family_config(use_mxfp8=True), context={axis: True})
        with self.assertRaisesRegex(NotImplementedError, "mlp.experts.*EP=1, 2 or 4"):
            replace_experts(TinyMoE(), family_config(use_mxfp8=True), context={"ep": True, "ep_size": 8})

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_parent_child_and_alias_conflicts(self):
        """
        Feature: GroupedExperts module family.
        Description: The real compiler rejects duplicate/nested replacement before construction.
        Expectation: Accepted contracts hold and unsupported requests fail explicitly.
        """
        config = family_config(use_mxfp8=True)
        config.plan_overrides.append(PlanOverride(match="mlp", module_type="torch.nn.Module",
                                                  replace_module=config.plan_overrides[0].replace_module))
        with self.assertRaisesRegex(ValueError, "descendant"):
            replace_experts(TinyMoE(), config)
        config = family_config()
        config.plan_overrides.append(copy.deepcopy(config.plan_overrides[0]))
        with self.assertRaisesRegex(ValueError, "conflict"):
            replace_experts(TinyMoE(), config)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_grouped_tristate_and_module_call(self):
        """
        Feature: GroupedExperts module family.
        Description: Parsed factory defaults, explicit True/False and experts hooks keep their meaning.
        Expectation: Accepted contracts hold and unsupported requests fail explicitly.
        """
        factory_path = "hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel.qwen3moe_ep_compute_fn"
        for low in (False, True):
            for requested in (None, True, False):
                with self.subTest(low=low, requested=requested):
                    model = TinyMoE()
                    replace_experts(model, family_config(use_mxfp8=low))
                    raw = family_config().to_dict()
                    args = {} if requested is None else {"use_grouped_gemm": requested}
                    raw["plan_overrides"].append({"match": "mlp", "when": "ep", "region_dispatch": False,
                                                  "local_compute_fn": {"_target_": factory_path, **args}})
                    target = resolve_config(raw).plan_overrides[-1].local_compute_fn
                    self.assertIs(target.use_grouped_gemm, requested)
                    if low and requested is False:
                        with self.assertRaisesRegex(ValueError, "mlp.experts.*explicit False"):
                            target.build(module=model.mlp, mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=FakeEpMesh())
                        continue
                    target.build(module=model.mlp, mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=FakeEpMesh())
                    self.assertIs(model.mlp.experts.ep_use_grouped_gemm, requested is not False)
                    if low or requested is False:
                        experts = model.mlp.experts
                        experts.gate_up_proj = nn.Parameter(experts.gate_up_proj[:2].detach())
                        experts.down_proj = nn.Parameter(experts.down_proj[:2].detach())
                        calls = []
                        experts.register_forward_hook(lambda *args: calls.append(1))
                        x = torch.randn(5, 128, requires_grad=True)
                        ids = torch.tensor([1, 0, 1, 0, 1])
                        actual = experts(x, ids)
                        expected = []
                        for token, expert_id in zip(x, ids):
                            gate, up = (token @ experts.gate_up_proj[expert_id]).chunk(2)
                            expected.append((torch.nn.functional.silu(gate) * up) @ experts.down_proj[expert_id])
                        torch.testing.assert_close(actual, torch.stack(expected))
                        actual.sum().backward()
                        self.assertEqual(calls, [1])
                        self.assertEqual(experts.local_expert_count, 2)
                        self.assertIsNotNone(x.grad)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_grouped_binder_supported_expert_sizes(self):
        """
        Feature: Grouped expert recipe capability validation.
        Description: Build the selected MXFP8 recipe for EP sizes two, four and eight.
        Expectation: Supported sizes derive local experts; EP=8 fails with the module FQN.
        """
        factory_path = "hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel.qwen3moe_ep_compute_fn"
        raw = family_config(use_mxfp8=True).to_dict()
        raw["plan_overrides"].append({"match": "mlp", "local_compute_fn": {"_target_": factory_path}})
        config = resolve_config(raw)
        target = config.plan_overrides[-1].local_compute_fn
        for size in (2, 4, 8):
            model, mesh = TinyMoE(), FakeEpMesh()
            replace_experts(model, config)
            with self.subTest(ep_size=size), patch.object(mesh, "size", return_value=size):
                if size == 8:
                    with self.assertRaisesRegex(NotImplementedError, "mlp.experts.*EP=1, 2 or 4"):
                        target.build(module=model.mlp, mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=mesh)
                else:
                    target.build(module=model.mlp, mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=mesh)
                    self.assertEqual(model.mlp.experts.local_expert_count, 4 // size)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_local_contract_and_gradients(self):
        """
        Feature: GroupedExperts module family.
        Description: Identity device arithmetic validates differentiable transposes and score ownership.
        Expectation: Accepted contracts hold and unsupported requests fail explicitly.
        """
        model = TinyMoE()
        replace_experts(model, family_config(use_mxfp8=True))
        experts = model.mlp.experts
        counts = torch.tensor([2, 0, 1, 3])
        x = torch.randn(6, 128, requires_grad=True)
        scores = torch.rand(6, requires_grad=True)
        output = experts.forward_expert_major(x, counts, scores)
        ref = []
        for block, w1, w2 in zip(x.split(counts.tolist()), experts.gate_up_proj, experts.down_proj):
            gate, up = (block @ w1).chunk(2, -1)
            ref.append((torch.nn.functional.silu(gate) * up) @ w2)
        reference = torch.cat(ref)
        torch.testing.assert_close(output, reference)
        variables = (x, experts.gate_up_proj, experts.down_proj)
        actual = torch.autograd.grad(output.sum(), variables, retain_graph=True)
        repeated = torch.autograd.grad(output.sum(), variables)
        expected = torch.autograd.grad(reference.sum(), variables)
        for grad, again, wanted in zip(actual, repeated, expected):
            torch.testing.assert_close(grad, wanted)
            torch.testing.assert_close(grad, again, atol=0, rtol=0)
        self.assertIsNone(scores.grad)
        for bad in (torch.tensor([2, 4]), torch.tensor([2., 0., 1., 3.])):
            with self.assertRaises(ValueError):
                experts.forward_expert_major(x, bad)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_legacy_parallel_guard(self):
        """
        Feature: GroupedExperts module family.
        Description: The old global precision path remains protected.
        Expectation: Accepted contracts hold and unsupported requests fail explicitly.
        """
        with self.assertRaisesRegex(NotImplementedError, "TP=CP=EP=PP=1"):
            _apply_module_replacement_actions(
                TinyMoE(), weights_mapping=[], context={"low_precision": object(), "ep": True})

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_baseline_without_optional_backend(self):
        """
        Feature: GroupedExperts module family.
        Description: A fresh interpreter builds baseline with both optional backends unavailable.
        Expectation: Accepted contracts hold and unsupported requests fail explicitly.
        """
        code = """import sys
sys.modules['torch_npu'] = None
sys.modules['omni'] = None
from tests.common.grouped_experts_family import family_config, TinyMoE, replace_experts
model = TinyMoE()
replace_experts(model, family_config())
assert 'hyper_parallel.components.modules.mxfp8_grouped_swiglu' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False,
                                env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"})
        self.assertEqual(result.returncode, 0, result.stderr)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_plan_preflight(self):
        """
        Feature: GroupedExperts module family.
        Description: Only EP Shard(0) and production execution are admitted for this family.
        Expectation: Accepted contracts hold and unsupported requests fail explicitly.
        """
        model = TinyMoE()
        replace_experts(model, family_config(use_mxfp8=True))
        mesh = SimpleNamespace(mesh_dim_names=())
        ep_mesh = FakeEpMesh()
        spec = ModuleShardingSpec(params={"experts.gate_up_proj": {EP: Shard(0)},
                                          "experts.down_proj": {EP: Shard(0)}},
                                   local_compute_fn=lambda: None)
        spec._ep_size = 2
        plan = ShardingPlan(modules={"mlp": spec})
        validate_module_family_plan(model, plan, mesh, ep_mesh, validate_mode=False)
        with patch.object(ep_mesh, "size", return_value=8):
            with self.assertRaisesRegex(NotImplementedError, "mlp.experts.*EP=1, 2 or 4"):
                validate_module_family_plan(model, plan, mesh, ep_mesh, validate_mode=False)
        with self.assertRaisesRegex(NotImplementedError, "mlp.experts.*production"):
            validate_module_family_plan(model, plan, mesh, ep_mesh, validate_mode=True)
        for placement in (Shard(1), Replicate()):
            spec.params["experts.down_proj"] = {EP: placement}
            with self.subTest(placement=placement), self.assertRaisesRegex((ValueError, NotImplementedError),
                                                                          "mlp.experts"):
                validate_module_family_plan(model, plan, mesh, ep_mesh, validate_mode=False)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_meta_alias_and_source_rejections(self):
        """
        Feature: GroupedExperts module family.
        Description: Meta construction and aliases preserve identity; unsupported activations/layouts fail.
        Expectation: Accepted contracts hold and unsupported requests fail explicitly.
        """
        model = TinyMoE(canonical=True, device="meta")
        source = model.mlp.experts
        model.alias = source
        replace_experts(model, family_config(use_mxfp8=True))
        self.assertIs(model.alias, model.mlp.experts)
        self.assertIs(model.mlp.experts.gate_up_proj, source.gate_up_proj)
        self.assertEqual(model.alias.gate_up_proj.device.type, "meta")
        for scenario in ("activation", "2d", "bias"):
            model = TinyMoE(canonical=True)
            expert = model.mlp.experts
            if scenario == "activation":
                expert.config.hidden_act = "gelu"
            elif scenario == "2d":
                expert.gate_up_proj = nn.Parameter(expert.gate_up_proj.reshape(4 * 128, 512))
                expert.down_proj = nn.Parameter(expert.down_proj.reshape(4 * 256, 128))
            else:
                expert.bias1 = nn.Parameter(torch.zeros(4, 512))
                expert.bias2 = nn.Parameter(torch.zeros(4, 128))
            with self.subTest(scenario=scenario), self.assertRaisesRegex(ValueError, "mlp.experts"):
                replace_experts(model, family_config(use_mxfp8=True))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_python_callable_arguments_preserved(self):
        """
        Feature: Python replacement target arguments.
        Description: Execute a declared callback through the actual replacement entry.
        Expectation: Callable objects reach the factory without serialization to strings.
        """
        received = []

        @module_replacement
        def _record(*, module, module_fqn, context, callback):
            """Keep runtime context and configured callback values intact."""
            callback((module, module_fqn, context["sentinel"]))
            return module

        model = nn.Sequential(nn.Linear(2, 2))
        marker = object()
        entries = [PlanOverride(match="0", module_type="torch.nn.Linear",
                                replace_module=Target(_record, target_path="test.record", callback=received.append))]
        _apply_module_replacement_actions(model, entries_to_module_replacements(entries),
                                         weights_mapping=[], context={"sentinel": marker})
        self.assertEqual(received, [(model[0], "0", marker)])

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
"""CPU contracts for the unified policy/Family implementation and preserved baseline."""

import copy
import subprocess
import sys
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.components.modules.moe.grouped_experts.low_precision_grouped_experts import (
    GroupedExperts,
    LowPrecisionGroupedExperts,
    replace_fake_w4a8_grouped_experts,
    replace_mxfp8_grouped_experts,
    replace_w4a8_grouped_experts,
)
from hyper_parallel.components.quantization.config import LowPrecisionDtypeScheme
from hyper_parallel.components.quantization.functional.fake_w4a8_gmm_func import FakeW4A8GroupedLinear
from hyper_parallel.components.quantization.functional.hifloat8_gmm_func import HiFloat8GroupedLinear
from hyper_parallel.components.quantization.functional.mxfp8_gmm_func import MXFP8GroupedLinear
from hyper_parallel.components.quantization.functional.w4a8_gmm_func import W4A8GroupedLinear
from hyper_parallel.components.quantization.modules import GroupedExperts as PolicyGroupedExperts
from hyper_parallel.components.quantization.modules.replacement import replace_grouped_experts
from hyper_parallel.trainer.config.resolver import resolve_config
from tests.common.grouped_experts_family import TinyMoE, cpu_grouped_matmul
from tests.common.qat_family import install_family, qat_config


_MODULE = "hyper_parallel.components.modules.moe.grouped_experts.low_precision_grouped_experts."
_STRATEGIES = "hyper_parallel.components.quantization.functional.strategy_factory."


def _cpu_projection(inputs, weight, counts, strategy, group_list_type):
    """Keep autograd and replace only the device projection with dense CPU matmuls."""
    del strategy, group_list_type
    return torch.cat([part @ matrix.T for part, matrix in zip(inputs.split(counts.tolist()), weight)])


def _cpu_baseline_projection(inputs, weight, *, group_list, **kwargs):
    """Adapt the ordinary grouped-GEMM signature to the shared CPU oracle."""
    del kwargs
    return cpu_grouped_matmul(inputs, weight, group_list=group_list)


def _hif8_config():
    """Select HiF8 through the real named-scheme resolver and Family rule."""
    raw = qat_config().to_dict()
    raw["training"]["low_precision"]["dtype_schemes"]["experts"] = {
        "weight_format": "hif8", "act_format": "hif8",
    }
    return resolve_config(raw)


def _unaligned_model() -> TinyMoE:
    """Use expert dimensions that native HiF8 supports without an MX block gate."""
    model = TinyMoE()
    source = model.mlp.experts
    source.hidden_size, source.intermediate_size = 31, 7
    source.gate_up_proj = nn.Parameter(torch.randn(4, 14, 31, dtype=torch.float64) * 0.1)
    source.down_proj = nn.Parameter(torch.randn(4, 31, 7, dtype=torch.float64) * 0.1)
    source.act_fn = nn.SiLU()
    return model


class TestUnifiedLowPrecisionGroupedExperts(unittest.TestCase):
    """Exercise public policy conversion and real Family selection using one class."""

    def setUp(self) -> None:
        """Keep randomness isolated and replace hardware probes only."""
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        for name in ("validate_npu_gmm_runtime", "validate_w4a8_gmm_runtime",
                     "validate_fake_w4a8_gmm_runtime", "validate_hifloat8_gmm_runtime"):
            probe = patch(_STRATEGIES + name)
            probe.start()
            self.addCleanup(probe.stop)

    def test_policy_and_hif8_family_share_type_with_distinct_layout_contracts(self) -> None:
        """
        Feature: Unified policy and HiF8 Family construction.
        Description: Resolve non-MX-aligned experts through both public entry points.
        Expectation: One class preserves policy registrations and Family checkpoint transforms.
        """
        model = _unaligned_model()
        source = model.mlp.experts
        source.register_buffer("persistent_state", torch.tensor(3))
        source.register_buffer("scratch_state", torch.tensor(5), persistent=False)
        source.extra = nn.Linear(2, 2)
        source.eval()
        before = source.gate_up_proj.detach().clone()
        rng = torch.get_rng_state()
        policy = replace_grouped_experts(
            module=source, module_fqn="mlp.experts",
            context={"low_precision": LowPrecisionDtypeScheme(weight_format="hif8", act_format="hif8")},
        )
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertIs(PolicyGroupedExperts, LowPrecisionGroupedExperts)
        self.assertIs(type(policy), LowPrecisionGroupedExperts)
        self.assertIs(policy.gate_up_proj, source.gate_up_proj)
        self.assertIs(policy.down_proj, source.down_proj)
        self.assertIs(policy.act_fn, source.act_fn)
        self.assertIs(policy.extra, source.extra)
        self.assertIs(policy.config, source.config)
        self.assertIs(policy.persistent_state, source.persistent_state)
        self.assertIs(policy.scratch_state, source.scratch_state)
        self.assertEqual(tuple(policy.state_dict()), tuple(source.state_dict()))
        self.assertNotIn("scratch_state", policy.state_dict())
        self.assertFalse(policy.training)
        self.assertEqual(policy.make_transforms(), [])

        family = install_family(model, _hif8_config()).mlp.experts
        self.assertIs(type(family), type(policy))
        self.assertIs(type(family.grouped_linear), HiFloat8GroupedLinear)
        self.assertEqual(tuple(policy.gate_up_proj.shape), (4, 14, 31))
        self.assertEqual(tuple(family.gate_up_proj.shape), (4, 31, 14))
        torch.testing.assert_close(family.gate_up_proj, before.transpose(-2, -1), rtol=0, atol=0)
        self.assertEqual(len(family.make_transforms()), 2)

    def test_both_layouts_share_expert_compute_and_weight_gradients(self) -> None:
        """
        Feature: Shared low-precision expert computation.
        Description: Execute canonical and transposed weights with an empty expert group.
        Expectation: Outputs, input gradients and both weight gradients agree at float64 precision.
        """
        model = _unaligned_model()
        policy = PolicyGroupedExperts.from_module(
            copy.deepcopy(model.mlp.experts), fqn="experts", grouped_linear=HiFloat8GroupedLinear(),
        )
        family = install_family(model, _hif8_config()).mlp.experts
        inputs = torch.randn(5, 31, dtype=torch.float64, requires_grad=True)
        family_inputs = inputs.detach().clone().requires_grad_()
        counts = torch.tensor([2, 0, 1, 2])
        with patch(_MODULE + "_GroupedLinearFunction.apply", side_effect=_cpu_projection):
            policy_out = policy.forward_expert_major(inputs, counts)
            family_out = family.forward_expert_major(family_inputs, counts)
        # CPU BLAS uses different kernels for the two strides; compare at float64 roundoff.
        torch.testing.assert_close(policy_out, family_out, rtol=1e-12, atol=1e-14)
        policy_grads = torch.autograd.grad(
            policy_out.square().sum(), (inputs, policy.gate_up_proj, policy.down_proj))
        family_grads = torch.autograd.grad(
            family_out.square().sum(), (family_inputs, family.gate_up_proj, family.down_proj))
        torch.testing.assert_close(policy_grads[0], family_grads[0], rtol=1e-12, atol=1e-14)
        for canonical, transposed in zip(policy_grads[1:], family_grads[1:]):
            torch.testing.assert_close(canonical, transposed.transpose(-2, -1), rtol=1e-12, atol=1e-14)

    def test_explicit_factories_keep_their_format_when_context_selects_hif8(self) -> None:
        """
        Feature: Explicit format factory compatibility.
        Description: Call each fixed-format factory with a different format in its context.
        Expectation: The factory selects its named strategy without mutating the caller's context.
        """
        context = {"low_precision": LowPrecisionDtypeScheme(weight_format="hif8", act_format="hif8")}
        cases = ((replace_mxfp8_grouped_experts, MXFP8GroupedLinear),
                 (replace_w4a8_grouped_experts, W4A8GroupedLinear),
                 (replace_fake_w4a8_grouped_experts, FakeW4A8GroupedLinear))
        for factory, expected in cases:
            with self.subTest(factory=factory.__name__):
                expert = factory(module=TinyMoE().mlp.experts, module_fqn="experts", context=context)
                self.assertIs(type(expert), LowPrecisionGroupedExperts)
                self.assertIs(type(expert.grouped_linear), expected)
                self.assertEqual(context["low_precision"].weight_format, "hif8")

    def test_dimension_constructor_retains_canonical_public_api(self) -> None:
        """
        Feature: Public quantization constructor compatibility.
        Description: Construct the exported alias with the existing dimension arguments.
        Expectation: The unified class retains canonical parameter shapes and target metadata.
        """
        expert = PolicyGroupedExperts(2, 4, 6, grouped_linear=HiFloat8GroupedLinear(), fqn="experts")
        self.assertIs(type(expert), LowPrecisionGroupedExperts)
        self.assertEqual(tuple(expert.gate_up_proj.shape), (2, 12, 4))
        self.assertEqual(tuple(expert.down_proj.shape), (2, 4, 6))
        self.assertEqual(expert.fqn, "experts")

    def test_public_policy_import_does_not_load_optional_npu_backend(self) -> None:
        """
        Feature: Optional NPU dependency isolation.
        Description: Import and construct the public policy adapter in a fresh process.
        Expectation: Canonical conversion preserves parameters without importing torch_npu.
        """
        script = textwrap.dedent("""
            import sys
            import torch
            from torch import nn
            sys.modules["torch_npu"] = None
            from hyper_parallel.components.quantization import GroupedExperts
            from hyper_parallel.components.quantization.functional.hifloat8_gmm_func import HiFloat8GroupedLinear
            from hyper_parallel.components.modules.moe.grouped_experts.low_precision_grouped_experts import (
                LowPrecisionGroupedExperts,
            )

            source = nn.Module()
            source.gate_up_proj = nn.Parameter(torch.ones(2, 6, 4))
            source.down_proj = nn.Parameter(torch.ones(2, 4, 3))
            converted = GroupedExperts.from_module(source, fqn="experts", grouped_linear=HiFloat8GroupedLinear())
            assert type(converted) is LowPrecisionGroupedExperts
            assert converted.gate_up_proj is source.gate_up_proj
            assert sys.modules["torch_npu"] is None
        """)
        result = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"exit={result.returncode}, stderr={result.stderr}")


class TestPreservedGroupedExpertsBaseline(unittest.TestCase):
    """Moving the baseline keeps its non-low-precision layouts and computation modes."""

    def setUp(self) -> None:
        """Restore random state after the dense oracle cases."""
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())

    @staticmethod
    def _source(*, flattened: bool, gated: bool, bias: bool, preweight: bool) -> nn.Module:
        source = nn.Module()
        source.num_experts, source.hidden_size, source.intermediate_size = 3, 4, 6
        source.config = SimpleNamespace(hidden_act="silu", use_fused_swiglu=False, initializer_range=0.02)
        source.has_gate = gated
        source.is_transposed = True
        source.router_gating_in_fp32 = not preweight
        width = 12 if gated else 6
        first = torch.randn(3, 4, width, dtype=torch.float64) * 0.1
        second = torch.randn(3, 6, 4, dtype=torch.float64) * 0.1
        setattr(source, "gate_up_proj" if gated else "up_proj",
                nn.Parameter(first.reshape(12, width) if flattened else first))
        source.down_proj = nn.Parameter(second.reshape(18, 4) if flattened else second)
        if bias:
            source.bias1 = nn.Parameter(torch.randn(3, width, dtype=torch.float64) * 0.1)
            source.bias2 = nn.Parameter(torch.randn(3, 4, dtype=torch.float64) * 0.1)
        return source

    def test_baseline_forward_and_backward_preserve_existing_modes(self) -> None:
        """
        Feature: Existing baseline expert contracts.
        Description: Exercise 2D storage, bias, ungated activation and pre-GMM2 routing scores.
        Expectation: Outputs and gradients match dense expert-major references for every mode.
        """
        cases = ((True, True, False, False), (False, True, True, False),
                 (False, False, False, False), (False, True, False, True), (True, False, True, True))
        for flattened, gated, bias, preweight in cases:
            with self.subTest(flattened=flattened, gated=gated, bias=bias, preweight=preweight):
                source = self._source(flattened=flattened, gated=gated, bias=bias, preweight=preweight)
                expert = GroupedExperts(module=source)
                inputs = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
                counts, scores = torch.tensor([2, 0, 1]), torch.tensor([0.2, 0.7, 0.3], dtype=torch.float64)
                first = expert.gate_up_proj if gated else expert.up_proj
                weights = first.view(3, 4, -1)
                down = expert.down_proj.view(3, 6, 4)
                expected_parts = []
                for index, (part, probabilities) in enumerate(zip(inputs.split([2, 0, 1]), scores.split([2, 0, 1]))):
                    projected = part @ weights[index]
                    if bias:
                        projected = projected + expert.bias1[index]
                    if gated:
                        gate, up = projected.chunk(2, dim=-1)
                        activated = F.silu(gate) * up
                    else:
                        activated = F.silu(projected)
                    if preweight:
                        activated = activated * probabilities[:, None]
                    result = activated @ down[index]
                    expected_parts.append(result + expert.bias2[index] if bias else result)
                expected = torch.cat(expected_parts)
                with patch(_MODULE + "grouped_matmul", side_effect=_cpu_baseline_projection):
                    actual = expert.forward_expert_major(inputs, counts, scores)
                torch.testing.assert_close(actual, expected)
                variables = (inputs, *expert.parameters())
                actual_grads = torch.autograd.grad(actual.square().sum(), variables)
                expected_grads = torch.autograd.grad(expected.square().sum(), variables)
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual_grad, expected_grad)
                self.assertEqual(expert.use_2d_experts, flattened)
                self.assertEqual(expert.add_bias, bias)
                self.assertIs(first, source.gate_up_proj if gated else source.up_proj)

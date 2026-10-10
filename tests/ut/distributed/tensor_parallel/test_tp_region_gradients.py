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
"""Autograd contracts for explicitly declared replicated TP region inputs."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed._builder.precompiled_boundary import PrecompiledBoundary
from hyper_parallel.distributed._builder.tp_region_gradients import region_gradient_operation
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec, TP
from tests.common.gqa_family import native_attention, replace_attention


class TPRegionGradientTests(unittest.TestCase):
    """Mock only the collective while keeping the actual boundary and autograd."""

    def test_planner_explicit_source_gradient_contract(self) -> None:
        """HSDP source reduction and standalone full gradients choose distinct adjoints."""
        model = torch.nn.Module()
        model.self_attn = replace_attention(native_attention("llama"))[0]
        model.config = model.self_attn.config
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",), rank_list=(0, 1), init_backend=False)
        for reduced in (False, True):
            plan = ShardingPlanner().plan(model, mesh, tp_size=2, source_gradient_reduction=reduced)
            self.assertIs(plan.modules["self_attn"]._tp_replicated_gradient_contract, not reduced)
            expected = "source_layout" if reduced else "replicated"
            self.assertEqual(plan.family_modules["self_attn"]["tp_gradient_contract"], expected)

    def test_identity_input_sums_backward_only(self) -> None:
        """Each rank receives the sum of local head contributions to input dgrad."""
        mesh = SimpleNamespace(get_group=lambda: "tp")
        operation = region_gradient_operation(mesh, ("tp",), (Replicate(),), (Replicate(),), input_side=True)
        value = torch.tensor([1., 2.], requires_grad=True)
        with patch("hyper_parallel.distributed._builder.tp_region_gradients.communication.differentiable_all_reduce",
                   side_effect=lambda tensor, _op, _group: tensor * 2) as collective:
            output = operation.execute(value)
            torch.testing.assert_close(output, value)
            collective.assert_not_called()
            output.sum().backward()
            collective.assert_called_once()
        torch.testing.assert_close(value.grad, torch.full_like(value, 2))

    def test_partial_output_does_not_sum_backward(self) -> None:
        """A replicated output contributes one logical loss gradient, not TP copies."""
        mesh = SimpleNamespace(get_group=lambda: "tp")
        operation = region_gradient_operation(mesh, ("tp",), (Partial(),), (Replicate(),), input_side=False)
        value = torch.tensor([1., 2.], requires_grad=True)
        with patch("hyper_parallel.distributed._builder.tp_region_gradients.communication.differentiable_all_reduce",
                   side_effect=lambda tensor, _op, _group: tensor * 2) as collective:
            output = operation.execute(value)
            torch.testing.assert_close(output, value * 2)
            output.sum().backward()
            collective.assert_called_once()
        torch.testing.assert_close(value.grad, torch.ones_like(value))

    def test_sequence_parallel_transitions_keep_existing_lowering(self) -> None:
        """SP all-gather and reduce-scatter retain their ordinary inverse adjoints."""
        mesh = SimpleNamespace(get_group=lambda: "tp")
        self.assertIsNone(region_gradient_operation(mesh, ("tp",), (Shard(1),), (Replicate(),), input_side=True))
        self.assertIsNone(region_gradient_operation(mesh, ("tp",), (Partial(),), (Shard(1),), input_side=False))

    def test_contract_is_opt_in_and_plan_explanation_needs_no_group(self) -> None:
        """Existing generic boundaries stay unchanged and offline explain remains safe."""
        mesh = SimpleNamespace(get_group=lambda: "tp")
        spec = ModuleShardingSpec(in_src={"hidden_states": {TP: Replicate()}},
                                  in_dst={"hidden_states": {TP: Replicate()}},
                                  out_src={"output": {TP: Partial()}},
                                  out_dst={"output": {TP: Replicate()}})
        plain = PrecompiledBoundary(spec, mesh, ("tp",))
        self.assertIsNone(plain.in_plan[0].execution_op)
        spec._tp_replicated_gradient_contract = True
        enabled = PrecompiledBoundary(spec, mesh, ("tp",))
        self.assertIsNotNone(enabled.in_plan[0].execution_op)
        self.assertIsNotNone(enabled.out_plan[0].execution_op)
        offline = PrecompiledBoundary(spec, None, ("tp",))
        self.assertIsNone(offline.in_plan[0].execution_op)

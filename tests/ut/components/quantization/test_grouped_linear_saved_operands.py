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
"""Physical saved operands across the shared grouped-linear lifecycle."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from hyper_parallel.components.quantization.functional.base_gmm_func import _GroupedLinearFunction
from hyper_parallel.components.quantization.tensor import MXFP8Tensor, W4A8Tensor
from tests.ut.components.quantization.functional.test_grouped_linear import _DenseGroupedLinear
from tests.common.mark_utils import arg_mark


class PhysicalGroupedLinear(_DenseGroupedLinear):
    """Isolate device arithmetic while retaining real wrappers and autograd."""

    def __init__(self, native_w4: bool = False) -> None:
        """Choose the directional contract to exercise."""
        self.native_w4 = native_w4

    def normalize_group_list(self, group_list: torch.Tensor, group_list_type: int) -> tuple[torch.Tensor, int]:
        """Normalize native W4 counts using the production convention."""
        if self.native_w4 and group_list_type == 1:
            return group_list.cumsum(0), 0
        return group_list, group_list_type

    def weight_quantization_directions(self, needs_grad_input: bool) -> tuple[bool, bool]:
        """Request the directions consumed by forward and dgrad."""
        return (True, needs_grad_input) if self.native_w4 else (needs_grad_input, True)

    def retain_weight_backward(self, weight_quant: Any, needs_grad_input: bool) -> None:
        """Release only the direction spent by forward."""
        weight_quant.update_usage(
            rowwise=needs_grad_input and not self.native_w4,
            colwise=needs_grad_input and self.native_w4,
        )

    def _store(self, value, rowwise, colwise):
        tensor_type = W4A8Tensor if self.native_w4 else MXFP8Tensor
        return tensor_type(
            value.shape, value.dtype, quantizer=None,
            row_data=value.detach().clone() if rowwise else None,
            row_scale=torch.ones(1) if rowwise else None,
            col_data=value.detach().clone() if colwise else None,
            col_scale=torch.ones(1) if colwise else None,
        )

    def grouped_matmul(self, left: Any, right: Any, **kwargs: Any) -> torch.Tensor:
        """Compute with CPU buffers instead of device quantized arithmetic."""
        def _physical(operand):
            value = operand.row_data if operand.row_data is not None else operand.col_data
            return SimpleNamespace(value=value)
        return super().grouped_matmul(_physical(left), _physical(right), **kwargs)


class TestGroupedLinearSavedOperands(unittest.TestCase):
    """Exercise retention and hooks independently of NPU arithmetic."""

    def setUp(self) -> None:
        """Keep test inputs deterministic without leaking RNG state."""
        self.rng = torch.random.get_rng_state()
        torch.manual_seed(7)

    def tearDown(self) -> None:
        """Restore the caller RNG state."""
        torch.random.set_rng_state(self.rng)

    @staticmethod
    def _operands(empty=False, grad_input=True, grad_weight=True):
        counts = torch.tensor([0, 0, 0] if empty else [2, 0, 3])
        inputs = torch.randn(0 if empty else 5, 4, requires_grad=grad_input)
        weight = torch.randn(3, 6, 4, requires_grad=grad_weight)
        return inputs, weight, counts

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_repeated_backward(self):
        """Feature: retention
        Description: two backwards
        Expectation: identical gradients."""
        for native in (False, True):
            with self.subTest(native_w4=native):
                inputs, weight, counts = self._operands()
                output = _GroupedLinearFunction.apply(inputs, weight, counts, PhysicalGroupedLinear(native), 1)
                first = torch.autograd.grad(output.sum(), (inputs, weight), retain_graph=True)
                second = torch.autograd.grad(output.sum(), (inputs, weight))
                for actual, expected in zip(second, first):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_source_group_version(self):
        """Feature: version check
        Description: mutate counts
        Expectation: backward rejects it."""
        for native in (False, True):
            for empty in (False, True):
                with self.subTest(native_w4=native, empty=empty):
                    inputs, weight, counts = self._operands(empty)
                    output = _GroupedLinearFunction.apply(inputs, weight, counts, PhysicalGroupedLinear(native), 1)
                    counts.add_(0)
                    with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
                        output.sum().backward()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_physical_hooks_frozen_empty_and_checkpoint(self):
        """Feature: saved hooks
        Description: clone physical storage
        Expectation: gradients preserved."""
        for native in (False, True):
            for empty in (False, True):
                for grad_input, grad_weight in ((True, True), (True, False), (False, True)):
                    with self.subTest(native_w4=native, empty=empty, gradients=(grad_input, grad_weight)):
                        inputs, weight, counts = self._operands(empty, grad_input, grad_weight)
                        strategy = PhysicalGroupedLinear(native)
                        def _run(x, w):
                            return _GroupedLinearFunction.apply(x, w, counts, strategy, 1)
                        targets = tuple(x for x in (inputs, weight) if x.requires_grad)
                        expected = torch.autograd.grad(_run(inputs, weight).sum(), targets)
                        saved = []
                        def _pack(tensor):
                            self.assertIs(type(tensor), torch.Tensor)
                            saved.append(tensor.shape)
                            return tensor.clone()
                        with torch.autograd.graph.saved_tensors_hooks(_pack, lambda tensor: tensor):
                            output = _run(inputs, weight)
                        self.assertTrue(saved)
                        actual = torch.autograd.grad(output.sum(), targets)
                        checkpoint_output = checkpoint(_run, inputs, weight, use_reentrant=False)
                        recomputed = torch.autograd.grad(checkpoint_output.sum(), targets)
                        for value, reference, ac_value in zip(actual, expected, recomputed):
                            torch.testing.assert_close(value, reference, rtol=0, atol=0)
                            torch.testing.assert_close(ac_value, reference, rtol=0, atol=0)

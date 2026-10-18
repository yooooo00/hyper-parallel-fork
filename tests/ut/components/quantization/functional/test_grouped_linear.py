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
"""CPU tests for grouped-linear strategy dispatch and shared autograd."""

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

import torch

from hyper_parallel.components.quantization.config import LowPrecisionDtypeScheme
from hyper_parallel.components.quantization.functional.base_gmm_func import (
    GroupedLinear,
    _GroupedLinearFunction,
    validate_grouped_linear_inputs,
)
from hyper_parallel.components.quantization.functional.fake_w4a8_gmm_func import (
    FakeW4A8GroupedLinear,
)
from hyper_parallel.components.quantization.functional.mxfp8_gmm_func import (
    MXFP8GroupedLinear,
)
from hyper_parallel.components.quantization.functional.strategy_factory import (
    build_low_precision_strategy,
)
from hyper_parallel.components.quantization.functional.w4a8_gmm_func import (
    W4A8GroupedLinear,
)
from tests.common.mark_utils import arg_mark


cpu_test = arg_mark(
    plat_marks=["cpu_linux", "cpu_macos"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)


class _DenseStorage:
    """Directional CPU storage implementing the saved-operand protocol."""

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype, *, quantizer: Any = None,
                 row_data: torch.Tensor | None = None, row_scale: torch.Tensor | None = None,
                 col_data: torch.Tensor | None = None, col_scale: torch.Tensor | None = None) -> None:
        """Retain the physical buffers used by the CPU arithmetic double."""
        self.shape, self.dtype, self.quantizer = shape, dtype, quantizer
        self.row_data, self.row_scale = row_data, row_scale
        self.col_data, self.col_scale = col_data, col_scale

    @property
    def value(self) -> torch.Tensor:
        """Return the retained arithmetic buffer."""
        return self.row_data if self.row_data is not None else self.col_data

    def update_usage(self, rowwise: bool = True, colwise: bool = True) -> None:
        """Release spent directional buffers."""
        if not rowwise:
            self.row_data = self.row_scale = None
        if not colwise:
            self.col_data = self.col_scale = None


class _DenseGroupedLinear(GroupedLinear):
    """High-precision test double that exercises the production lifecycle."""

    FORMAT_NAME = "DenseTest"

    def normalize_group_list(self, group_list: torch.Tensor, group_list_type: int) -> tuple[torch.Tensor, int]:
        """Keep the supplied grouping representation."""
        return group_list, group_list_type

    def output_features(self, weight: torch.Tensor) -> int:
        """Read the output width from the live weight layout."""
        return weight.shape[-2]

    def weight_quantization_directions(self, needs_grad_input: bool) -> tuple[bool, bool]:
        """Follow the MXFP8 directional contract."""
        return needs_grad_input, True

    def retain_weight_backward(self, weight_quant: _DenseStorage, needs_grad_input: bool) -> None:
        """Keep the weight direction needed by dgrad."""
        weight_quant.update_usage(rowwise=needs_grad_input, colwise=False)

    @staticmethod
    def _store(value, rowwise, colwise):
        return _DenseStorage(
            value.shape, value.dtype,
            row_data=value if rowwise else None, row_scale=torch.ones(1) if rowwise else None,
            col_data=value if colwise else None, col_scale=torch.ones(1) if colwise else None,
        )

    def quantize_input(self, inputs: torch.Tensor, *, rowwise: bool, colwise: bool, **kwargs: Any) -> _DenseStorage:
        """Store inputs without introducing device arithmetic."""
        del kwargs
        return self._store(inputs, rowwise, colwise)

    def quantize_weight(self, weight: torch.Tensor, *, rowwise: bool, colwise: bool) -> _DenseStorage:
        """Store weights under the same directional protocol."""
        return self._store(weight, rowwise, colwise)

    def quantize_grad_output(
        self, grad_output: torch.Tensor, *, rowwise: bool, colwise: bool, **kwargs: Any,
    ) -> _DenseStorage:
        """Store upstream gradients for dgrad and wgrad."""
        del kwargs
        return self._store(grad_output, rowwise, colwise)

    @staticmethod
    def _counts(group_list, group_list_type):
        if group_list_type == 1:
            return group_list.tolist()
        boundaries = group_list.tolist()
        return [end - (boundaries[index - 1] if index else 0)
                for index, end in enumerate(boundaries)]

    def grouped_matmul(
        self,
        left: _DenseStorage,
        right: _DenseStorage,
        *,
        layout: str,
        group_list: torch.Tensor,
        group_type: int,
        group_list_type: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Execute dense grouped arithmetic using the retained directions."""
        del group_type
        counts = self._counts(group_list, group_list_type)
        left_parts = left.value.split(counts, dim=0)
        if layout == "NN":
            result = [part @ right.value[index] for index, part in enumerate(left_parts)]
            return torch.cat(result, dim=0).to(output_dtype)
        if layout == "NT":
            result = [part @ right.value[index].transpose(-2, -1)
                      for index, part in enumerate(left_parts)]
            return torch.cat(result, dim=0).to(output_dtype)
        if layout == "TN":
            right_parts = right.value.split(counts, dim=0)
            result = [
                left_part.transpose(-2, -1) @ right_part
                for left_part, right_part in zip(left_parts, right_parts)
            ]
            return torch.stack(result, dim=0).to(output_dtype)
        raise AssertionError(f"unexpected layout: {layout}")


def _dense_reference(inputs, weight, counts):
    """Reference grouped linear for live [E, O, K] weights."""
    parts = inputs.split(counts.tolist(), dim=0)
    return torch.cat(
        [part @ weight[index].transpose(-2, -1) for index, part in enumerate(parts)],
        dim=0,
    )


class TestStrategyFactory(unittest.TestCase):
    """Policy-to-strategy dispatch without importing an NPU runtime."""

    @staticmethod
    def _patch_probes():
        module = "hyper_parallel.components.quantization.functional.strategy_factory"
        return (
            mock.patch(f"{module}.validate_npu_gmm_runtime"),
            mock.patch(f"{module}.validate_w4a8_gmm_runtime"),
            mock.patch(f"{module}.validate_fake_w4a8_gmm_runtime"),
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dispatches_w8a8_native_w4a8_and_fake_w4a8(self):
        """Feature: grouped linear.
        Description: All three supported policies choose their independent strategies.
        Expectation: declared output, gradients or errors are preserved.
        """
        aligned32 = ((2, 64, 32), (2, 32, 64))
        aligned128 = ((2, 256, 128), (2, 128, 256))
        p_mx, p_native, p_fake = self._patch_probes()
        with p_mx as mx_probe, p_native as native_probe, p_fake as fake_probe:
            w8a8 = build_low_precision_strategy(None, tile_shapes=aligned32)
            native = build_low_precision_strategy(
                LowPrecisionDtypeScheme(
                    weight_format="mxfp4", act_format="mxfp8", block_size=128
                ),
                tile_shapes=aligned128,
            )
            fake = build_low_precision_strategy(
                LowPrecisionDtypeScheme(
                    is_fake_quantize=True,
                    weight_format="mxfp4",
                    act_format="mxfp8",
                    block_size=32,
                ),
                tile_shapes=aligned32,
            )

        self.assertIsInstance(w8a8, MXFP8GroupedLinear)
        self.assertIsInstance(native, W4A8GroupedLinear)
        self.assertEqual(native.block_size, 128)
        self.assertIsInstance(fake, FakeW4A8GroupedLinear)
        mx_probe.assert_called_once_with()
        native_probe.assert_called_once_with()
        fake_probe.assert_called_once_with()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_alignment_and_unsupported_policies_fail_at_factory(self):
        """Feature: grouped linear.
        Description: Bad tiles and unimplemented family combinations fail before replacement.
        Expectation: declared output, gradients or errors are preserved.
        """
        module = "hyper_parallel.components.quantization.functional.strategy_factory"
        with mock.patch(f"{module}.validate_w4a8_gmm_runtime"):
            with self.assertRaisesRegex(ValueError, "not aligned"):
                build_low_precision_strategy(
                    LowPrecisionDtypeScheme(
                        weight_format="mxfp4", act_format="mxfp8"
                    ),
                    tile_shapes=((2, 63, 32), (2, 32, 64)),
                )
        with self.assertRaisesRegex(NotImplementedError, "fake QAT currently supports"):
            build_low_precision_strategy(
                LowPrecisionDtypeScheme(is_fake_quantize=True),
                tile_shapes=((2, 64, 32), (2, 32, 64)),
            )
        with self.assertRaisesRegex(NotImplementedError, "only the mxfp family"):
            build_low_precision_strategy(
                LowPrecisionDtypeScheme(weight_format="hif8", act_format="hif8"),
                tile_shapes=((2, 64, 32), (2, 32, 64)),
            )


class TestGroupedLinearLifecycle(unittest.TestCase):
    """The shared autograd bridge preserves dense forward/dgrad/wgrad semantics."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_forward_and_both_gradients_match_dense_reference(self):
        """Feature: grouped linear.
        Description: Live [E,O,K] weights are transposed only at the GMM seam.
        Expectation: declared output, gradients or errors are preserved.
        """
        torch.manual_seed(7)
        counts = torch.tensor([2, 3], dtype=torch.int64)
        inputs = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
        weight = torch.randn(2, 6, 4, dtype=torch.float64, requires_grad=True)
        grad = torch.randn(5, 6, dtype=torch.float64)

        actual = _GroupedLinearFunction.apply(
            inputs, weight, counts, _DenseGroupedLinear(), 1
        )
        actual_grads = torch.autograd.grad(actual, (inputs, weight), grad)

        ref_inputs = inputs.detach().clone().requires_grad_(True)
        ref_weight = weight.detach().clone().requires_grad_(True)
        expected = _dense_reference(ref_inputs, ref_weight, counts)
        expected_grads = torch.autograd.grad(expected, (ref_inputs, ref_weight), grad)

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_grads[0], expected_grads[0])
        torch.testing.assert_close(actual_grads[1], expected_grads[1])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_empty_input_has_shaped_output_and_zero_gradients(self):
        """Feature: grouped linear.
        Description: Experts receiving zero total tokens remain valid in forward/backward.
        Expectation: declared output, gradients or errors are preserved.
        """
        inputs = torch.empty(0, 4, requires_grad=True)
        weight = torch.randn(2, 6, 4, requires_grad=True)
        counts = torch.tensor([0, 0], dtype=torch.int64)

        output = _GroupedLinearFunction.apply(
            inputs, weight, counts, _DenseGroupedLinear(), 1
        )
        output.sum().backward()

        self.assertEqual(tuple(output.shape), (0, 6))
        torch.testing.assert_close(inputs.grad, torch.zeros_like(inputs))
        torch.testing.assert_close(weight.grad, torch.zeros_like(weight))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_shared_shape_and_group_validation(self):
        """Feature: grouped linear.
        Description: Malformed geometry is rejected consistently for every strategy.
        Expectation: declared output, gradients or errors are preserved.
        """
        inputs = torch.randn(3, 4)
        weight = torch.randn(2, 6, 4)
        groups = torch.tensor([1, 2])
        validate_grouped_linear_inputs(
            inputs, weight, groups, 1, weight_input_dim=-1, name="test"
        )
        with self.assertRaisesRegex(ValueError, "contracting dimensions differ"):
            validate_grouped_linear_inputs(
                inputs, torch.randn(2, 6, 5), groups, 1,
                weight_input_dim=-1, name="test"
            )
        with self.assertRaisesRegex(ValueError, "one group per expert"):
            validate_grouped_linear_inputs(
                inputs, weight, torch.tensor([3]), 1,
                weight_input_dim=-1, name="test"
            )


if __name__ == "__main__":
    unittest.main()

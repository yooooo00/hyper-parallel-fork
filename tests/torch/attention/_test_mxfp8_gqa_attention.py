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
"""A5 GQA projection acceptance using the existing local MXFP8 error bounds.

Before execution, output/dX/dW require NRMSE <= .10 and maximum absolute
error <= .5 * reference maximum + 1e-6, as in grouped-expert MXFP8 tests.
Native versus BF16 baseline is reported separately against rtol=.01,
atol=.0001; an unmet native criterion is not a passing MXFP8 validation.
These local criteria do not establish language-model convergence.
"""

import copy
import json
import os
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from hyper_parallel.components.functional import apply_rotary_pos_emb
from hyper_parallel.components.quantization.functional.npu_mxfp8 import MXFP8NpuOps
from hyper_parallel.components.quantization.quantizers.mxfp8 import MXFP8Quantizer
from tests.common.gqa_family import attention_inputs, native_attention, replace_attention


class CountingLinearOps(MXFP8NpuOps):
    """Observe real MXFP8 dense matmuls without changing arithmetic."""

    def __init__(self, backend: Any) -> None:
        """Bind the real backend and reset dense call observations."""
        super().__init__(backend)
        self.calls = 0

    def quant_matmul(self, x1: torch.Tensor, x2: torch.Tensor,
                     x2_scale: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Require true FP8 operands then invoke the existing device adapter.

        Args:
            x1: Quantized activation operand passed to the device matmul.
            x2: Quantized projection-weight operand passed to the device matmul.
            x2_scale: Block scales associated with the projection weights.
            kwargs: Keyword arguments forwarded unchanged to the observed operation.
        """
        assert x1.dtype == x2.dtype == torch.float8_e4m3fn
        self.calls += 1
        return super().quant_matmul(x1, x2, x2_scale, **kwargs)


def _device() -> tuple[torch.device, Any]:
    backend = pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Real A5 hardware is required for MXFP8 projection numerics")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    return torch.device("npu", local_rank), backend


def _metric(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    error = (actual.float() - expected.float()).abs()
    rms = expected.float().square().mean().sqrt().item()
    error_rms = error.square().mean().sqrt().item()
    maximum = error.max().item()
    record = {"name": name, "nrmse": error_rms / max(rms, 1e-12), "max_abs": maximum}
    print(json.dumps(record), flush=True)
    assert torch.isfinite(actual).all()
    assert error_rms <= .10 * rms + 1e-6, record
    assert maximum <= .5 * expected.float().abs().max().item() + 1e-6, record


def _result(module: torch.nn.Module, inputs: dict) -> tuple[torch.Tensor, dict]:
    hidden = inputs["hidden_states"].detach().clone().requires_grad_()
    output = module(**{**inputs, "hidden_states": hidden})[0]
    names, params = zip(*module.named_parameters())
    gradients = torch.autograd.grad(output.float().square().sum(), (hidden, *params), retain_graph=True)
    repeated = torch.autograd.grad(output.float().square().sum(), (hidden, *params))
    for actual, expected in zip(gradients, repeated):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return output, dict(zip(("input", *names), gradients))


def _observe_native(baseline: torch.nn.Module, actual: tuple, expected: tuple, prefix: str) -> None:
    """Record the unaligned native implementation without changing acceptance."""
    gradients = dict(actual[1])
    for transform in baseline.make_transforms():
        gradients.update(transform.operations[0].reverse_op.convert(
            gradients, transform.target_patterns, transform.source_patterns))
    for name, reference in {"output": expected[0], **expected[1]}.items():
        value = actual[0] if name == "output" else gradients[name]
        difference = (value.float() - reference.float()).abs()
        print(json.dumps({"name": f"{prefix}/{name}", "max_abs": difference.max().item(),
                          "pointwise_rtol": .01, "pointwise_atol": .0001,
                          "pointwise_pass": torch.allclose(value, reference, rtol=.01, atol=.0001),
                          "nrmse": (difference.square().mean().sqrt()
                                    / reference.float().square().mean().sqrt().clamp_min(1e-12)).item()}), flush=True)


@pytest.mark.parametrize("kind", ["llama", "qwen3_moe"])
def test_projection_numerics(kind: str) -> None:
    """Native/baseline/MXFP8 output, dX/dW, repetition and optimizer updates.

    Args:
        kind: Native attention protocol: Llama or Qwen3-MoE.
    """
    device, backend = _device()
    torch.manual_seed(793)
    native = native_attention(kind, device=str(device), bias=True)
    baseline, _ = replace_attention(copy.deepcopy(native))
    low, _ = replace_attention(copy.deepcopy(native), use_mxfp8=True)
    counter = CountingLinearOps(backend)
    for projection in (low.linear_qkv, low.o_proj):
        projection.quantizer = MXFP8Quantizer(npu_ops=counter)
    inputs = attention_inputs(device=str(device))
    original = _result(native, inputs)
    reference = _result(baseline, inputs)
    _observe_native(baseline, reference, original, "original_native")
    with patch(f"{type(native).__module__}.apply_rotary_pos_emb", side_effect=apply_rotary_pos_emb):
        expected = _result(native, inputs)
    actual = _result(low, inputs)
    _observe_native(baseline, reference, expected, "native_with_same_rotary_kernel")
    _metric(f"{kind}/output", actual[0], reference[0])
    for name, gradient in actual[1].items():
        _metric(f"{kind}/{name}", gradient, reference[1][name])
    assert counter.calls == 10
    optimizer = torch.optim.AdamW(low.parameters(), lr=.001)
    initial = low.linear_qkv.weight.detach().clone()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = low(**inputs)[0]
        loss = output.float().square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert low.linear_qkv.weight.grad is not None
        optimizer.step()
    assert not torch.equal(initial, low.linear_qkv.weight)
    print(json.dumps({"model": kind, "fp8_matmul_calls": counter.calls, "optimizer_steps": 3}), flush=True)


@pytest.mark.parametrize("kind", ["llama", "qwen3_moe"])
def test_projection_saved_tensors(kind: str) -> None:
    """Frozen projections and physical saved tensors survive offload/recompute.

    Args:
        kind: Native attention protocol: Llama or Qwen3-MoE.
    """
    device, _ = _device()
    torch.manual_seed(862)
    low, _ = replace_attention(native_attention(kind, device=str(device)), use_mxfp8=True)
    low.o_proj.weight.requires_grad_(False)
    inputs = attention_inputs(device=str(device))
    values = inputs.pop("hidden_states").requires_grad_()
    targets = (values, *[param for param in low.parameters() if param.requires_grad])
    direct = low(values, **inputs)[0]
    expected = torch.autograd.grad(direct.float().square().sum(), targets)
    with torch.autograd.graph.save_on_cpu():
        recomputed = checkpoint(lambda hidden: low(hidden, **inputs)[0], values, use_reentrant=False)
    actual = torch.autograd.grad(recomputed.float().square().sum(), targets, retain_graph=True)
    again = torch.autograd.grad(recomputed.float().square().sum(), targets)
    torch.testing.assert_close(recomputed, direct, rtol=0, atol=0)
    for gradient, repeated, reference in zip(actual, again, expected):
        torch.testing.assert_close(gradient, repeated, rtol=0, atol=0)
        torch.testing.assert_close(gradient, reference, rtol=0, atol=0)
    assert low.o_proj.weight.grad is None


@pytest.mark.parametrize("kind", ["llama", "qwen3_moe"])
def test_projection_empty_input(kind: str) -> None:
    """Empty batches retain every zero gradient without calling FP8 kernels.

    Args:
        kind: Native attention protocol: Llama or Qwen3-MoE.
    """
    device, backend = _device()
    for batch, sequence in ((2, 0), (0, 32)):
        for low in (False, True):
            module, _ = replace_attention(native_attention(kind, device=str(device), bias=True), use_mxfp8=low)
            counter = CountingLinearOps(backend)
            if low:
                for projection in (module.linear_qkv, module.o_proj):
                    projection.quantizer = MXFP8Quantizer(npu_ops=counter)
            hidden = torch.empty(batch, sequence, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
            output, weights = module(hidden)
            assert output.shape == hidden.shape
            assert weights is None
            gradients = torch.autograd.grad(output.sum(), (hidden, *module.parameters()))
            for gradient in gradients:
                assert torch.count_nonzero(gradient) == 0
            assert counter.calls == 0

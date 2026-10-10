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
"""Loaded/meta GQA projection contracts through the real replacement executor."""

import copy
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from hyper_parallel.components.modules.gqa_attention import GQAAttention
from hyper_parallel.components.modules.mxfp8_gqa_attention import MXFP8GQAAttention
from hyper_parallel.components.quantization.modules.mxfp8_linear import MXFP8Linear
from tests.common.gqa_family import attention_inputs, native_attention, replace_attention


class MXFP8GQAAttentionTests(unittest.TestCase):
    """Mock only hardware capability/arithmetic, never Family selection."""

    def setUp(self) -> None:
        """Keep CPU tests independent of the optional device extension."""
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(793)
        runtime = patch("hyper_parallel.components.modules.mxfp8_gqa_attention.validate_npu_runtime")
        runtime.start()
        self.addCleanup(runtime.stop)
        rotary = patch("hyper_parallel.components.modules.gqa_attention.apply_rotary_pos_emb",
                       side_effect=apply_rotary_pos_emb)
        rotary.start()
        self.addCleanup(rotary.stop)

    def test_loaded_weight_conversion_and_inverse(self) -> None:
        """Both native classes retain weight/bias values, freezing and QKV order."""
        for kind in ("llama", "qwen3_moe"):
            for low in (False, True):
                with self.subTest(model=kind, low=low):
                    source = native_attention(kind, bias=True).eval()
                    for projection in (source.q_proj, source.k_proj, source.v_proj):
                        projection.weight.requires_grad_(False)
                    original = copy.deepcopy(source.state_dict())
                    replacement, _ = replace_attention(source, use_mxfp8=low)
                    self.assertIs(type(replacement), MXFP8GQAAttention if low else GQAAttention)
                    self.assertFalse(replacement.training)
                    self.assertFalse(replacement.linear_qkv.weight.requires_grad)
                    self.assertIs(replacement.o_proj.weight, source.o_proj.weight)
                    self.assertEqual(replacement.linear_qkv.weight.dtype, torch.bfloat16)
                    self.assertEqual(replacement.linear_qkv.weight.shape, (256, 128))
                    for transform in replacement.make_transforms():
                        reverse = transform.operations[0].reverse_op.convert(
                            replacement.state_dict(), transform.target_patterns, transform.source_patterns)
                        for name, value in reverse.items():
                            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
                    qkv = replacement.linear_qkv.weight.view(2, 128, 128)
                    torch.testing.assert_close(qkv[:, :64].reshape(128, 128), original["q_proj.weight"])
                    torch.testing.assert_close(qkv[:, 64:96].reshape(64, 128), original["k_proj.weight"])
                    torch.testing.assert_close(qkv[:, 96:].reshape(64, 128), original["v_proj.weight"])

    def test_native_baseline_forward_and_gradients(self) -> None:
        """Float32 attention and all gradients agree for two actual model classes."""
        for kind in ("llama", "qwen3_moe"):
            with self.subTest(model=kind):
                native = native_attention(kind, dtype=torch.float32, bias=True)
                baseline, _ = replace_attention(copy.deepcopy(native))
                inputs = attention_inputs(sequence=7, dtype=torch.float32)
                native_x = inputs.pop("hidden_states").requires_grad_()
                baseline_x = native_x.detach().clone().requires_grad_()
                expected = native(native_x, **inputs)[0]
                actual = baseline(baseline_x, **inputs)[0]
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                gradient = torch.randn_like(expected)
                expected.backward(gradient)
                actual.backward(gradient)
                torch.testing.assert_close(baseline_x.grad, native_x.grad, rtol=1e-5, atol=1e-6)
                for transform in baseline.make_transforms():
                    target_grads = {name: baseline.get_parameter(name).grad for name in transform.target_patterns}
                    source_grads = transform.operations[0].reverse_op.convert(
                        target_grads, transform.target_patterns, transform.source_patterns)
                    for name, value in source_grads.items():
                        torch.testing.assert_close(value, native.get_parameter(name).grad, rtol=1e-5, atol=1e-6)

    def test_meta_materialization_and_state_dict_roundtrip(self) -> None:
        """Checkpoint conversion fills meta parameters and restores canonical values."""
        for kind in ("llama", "qwen3_moe"):
            for low in (False, True):
                with self.subTest(model=kind, low=low):
                    source = native_attention(kind, bias=True)
                    target, _ = replace_attention(native_attention(kind, device="meta", bias=True), use_mxfp8=low)
                    self.assertTrue(target.linear_qkv.weight.is_meta)
                    canonical = dict(source.state_dict())
                    for transform in target.make_transforms():
                        canonical.update(transform.operations[0].convert(
                            canonical, transform.source_patterns, transform.target_patterns))
                        for name in transform.source_patterns:
                            del canonical[name]
                    target.load_state_dict(canonical, assign=True)
                    saved = copy.deepcopy(target.state_dict())
                    restored, _ = replace_attention(native_attention(kind, device="meta", bias=True), use_mxfp8=low)
                    restored.load_state_dict(saved, assign=True)
                    for name, value in restored.state_dict().items():
                        self.assertFalse(value.is_meta)
                        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
                    for transform in restored.make_transforms():
                        original = transform.operations[0].reverse_op.convert(
                            saved, transform.target_patterns, transform.source_patterns)
                        for name, value in original.items():
                            torch.testing.assert_close(value, source.state_dict()[name], rtol=0, atol=0)

    def test_fused_source_projection_roundtrip(self) -> None:
        """A native contiguous Q|K|V source uses the same reversible grouping."""
        for low in (False, True):
            with self.subTest(low=low):
                separate = native_attention("llama", bias=True)
                source = nn.Module()
                source.config = separate.config
                source.o_proj = separate.o_proj
                source.qkv_proj = nn.Linear(128, 256, dtype=torch.bfloat16)
                with torch.no_grad():
                    for name in ("weight", "bias"):
                        packed = torch.cat([getattr(getattr(separate, proj), name)
                                            for proj in ("q_proj", "k_proj", "v_proj")])
                        getattr(source.qkv_proj, name).copy_(packed)
                constructor = MXFP8GQAAttention if low else GQAAttention
                target = constructor(module=source)
                for transform in target.make_transforms():
                    restored = transform.operations[0].reverse_op.convert(
                        target.state_dict(), transform.target_patterns, transform.source_patterns)
                    for name, value in restored.items():
                        torch.testing.assert_close(value, source.get_parameter(name), rtol=0, atol=0)

    def test_incompatible_source_projection_properties_fail(self) -> None:
        """Mixed QKV precision or trainability cannot be represented by one parameter."""
        source = native_attention("llama")
        source.k_proj.weight.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "cannot be represented"):
            replace_attention(source)
        source.k_proj.weight.requires_grad_(True)
        source.k_proj.to(dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "cannot be represented"):
            replace_attention(source)

    def test_projection_hooks_and_forward_are_preserved(self) -> None:
        """Existing output hooks and new QKV hooks execute through inherited forward."""
        source = native_attention("llama", bias=True)
        calls = []
        handle = source.o_proj.register_forward_hook(lambda *_: calls.append("output"))
        replacement, _ = replace_attention(source, use_mxfp8=True)
        replacement.linear_qkv.register_forward_pre_hook(lambda *_: calls.append("input"))
        self.assertIsInstance(replacement.linear_qkv, MXFP8Linear)
        self.assertIsInstance(replacement.o_proj, MXFP8Linear)
        self.assertIs(MXFP8GQAAttention.forward, GQAAttention.forward)
        self.assertIs(MXFP8GQAAttention.make_transforms, GQAAttention.make_transforms)
        with patch("hyper_parallel.components.quantization.modules.mxfp8_linear.mxfp8_linear",
                   side_effect=lambda inputs, weight, _quantizer: F.linear(inputs, weight)):  # pylint: disable=not-callable
            replacement(**attention_inputs(sequence=3))
            self.assertEqual(calls, ["input", "output"])
            handle.remove()
            replacement(**attention_inputs(sequence=3))
            self.assertEqual(calls, ["input", "output", "input"])

    def test_unsupported_dtype_and_unaligned_projection_report_path(self) -> None:
        """Unvalidated precision and tile shapes fail before any device compute."""
        with self.assertRaisesRegex(ValueError, "attention.*BF16"):
            replace_attention(native_attention("llama", dtype=torch.float16), use_mxfp8=True)
        source = native_attention("llama")
        source.o_proj = nn.Linear(128, 127, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "attention.*o_proj.*divisible by 32"):
            replace_attention(source, use_mxfp8=True)

    def test_empty_input_preserves_hooks_and_zero_gradients(self) -> None:
        """No device computation is necessary, while all empty graph edges remain."""
        for low in (False, True):
            with self.subTest(low=low):
                module, _ = replace_attention(native_attention("qwen3_moe", bias=True), use_mxfp8=low)
                module.attention_interface = Mock(side_effect=AssertionError("empty attention kernel called"))
                calls = []
                module.linear_qkv.register_forward_hook(lambda *_: calls.append("qkv"))
                module.o_proj.register_forward_hook(lambda *_: calls.append("output"))
                hidden = torch.empty(2, 0, 128, dtype=torch.bfloat16, requires_grad=True)
                with patch("hyper_parallel.components.quantization.modules.mxfp8_linear.mxfp8_linear") as compute:
                    output, _ = module(hidden)
                gradients = torch.autograd.grad(output.sum(), (hidden, *module.parameters()))
                compute.assert_not_called()
                module.attention_interface.assert_not_called()
                self.assertEqual(calls, ["qkv", "output"])
                for gradient in gradients:
                    self.assertEqual(torch.count_nonzero(gradient), 0)

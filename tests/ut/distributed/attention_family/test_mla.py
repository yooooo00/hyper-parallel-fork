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
"""Loaded and meta latent-projection contracts through actual MLA replacement."""

import copy
import unittest

import torch
from torch import nn

from hyper_parallel.models.replacement import _apply_module_replacement_actions
from hyper_parallel.trainer.config import entries_to_module_replacements
from tests.torch.context_parallel._test_mla_family import _MLAModel, _config


class MLAFamilyWeightsTests(unittest.TestCase):
    """Ensure the CP adapter starts from source-equivalent projection weights."""

    def test_loaded_and_meta_latent_projection_values(self):
        """Preserve packed projection values, bias, dtype, freezing and inverse conversion."""
        for device in ("cpu", "meta"):
            for bias in (False, True):
                with self.subTest(device=device, bias=bias):
                    with torch.device(device):
                        model = _MLAModel()
                        source = model.self_attn
                        for name in ("q_a_proj", "kv_a_proj_with_mqa"):
                            previous = getattr(source, name)
                            projection = nn.Linear(previous.in_features, previous.out_features, bias=bias)
                            projection.requires_grad_(False)
                            setattr(source, name, projection)
                    original = copy.deepcopy(source.state_dict())
                    config = _config(False)
                    mapping = []
                    _apply_module_replacement_actions(
                        model, entries_to_module_replacements(config.plan_overrides), weights_mapping=mapping,
                    )
                    replacement = model.self_attn
                    self.assertEqual(replacement.linear_qkv.weight.device.type, device)
                    self.assertFalse(replacement.linear_qkv.weight.requires_grad)
                    self.assertEqual(replacement.linear_qkv.bias is not None, bias)
                    if bias:
                        self.assertFalse(replacement.linear_qkv.bias.requires_grad)
                    if device == "meta":
                        self.assertTrue(replacement.linear_qkv.weight.is_meta)
                        continue
                    for transform in replacement.make_transforms():
                        restored = transform.operations[0].reverse_op.convert(
                            replacement.state_dict(), transform.target_patterns, transform.source_patterns,
                        )
                        for name, value in restored.items():
                            torch.testing.assert_close(value, original[name], atol=0, rtol=0)

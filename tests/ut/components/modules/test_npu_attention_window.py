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
"""CPU contracts for rectangular attention mask and window forwarding."""

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.components.functional import npu_fusion_attention_forward


class NPUAttentionWindowTests(unittest.TestCase):
    """Mock only the device operator; retain public argument normalization."""

    def _invoke(self, query_length, key_length, mask=None, module=None, **kwargs):
        """Capture the real interface's fully normalized operator arguments."""
        query = torch.zeros(1, 4, query_length, 32)
        key = torch.zeros(1, 2, key_length, 32)
        if module is None:
            module = torch.nn.Module()
        with patch("torch_npu.npu_fusion_attention", return_value=(query,)) as kernel:
            output, weights = npu_fusion_attention_forward(module, query, key, key, mask, **kwargs)
        self.assertEqual(output.shape, (1, query_length, 4, 32))
        self.assertIsNone(weights)
        return kernel.call_args.kwargs

    def test_default_window_retains_all_globally_allowed_keys(self):
        """CP or cache queries must not receive an extra local causal window."""
        for query_length, key_length in ((48, 96), (64, 128), (1, 96), (64, 64)):
            with self.subTest(query=query_length, key=key_length):
                arguments = self._invoke(query_length, key_length)
                expected_next = 0 if query_length == key_length else 1048576
                self.assertEqual(arguments["next_tockens"], expected_next)
                query_positions = torch.arange(query_length) + key_length - query_length
                blocked = torch.arange(key_length)[None, :] > query_positions[:, None]
                torch.testing.assert_close(arguments["atten_mask"], blocked)

    def test_explicit_masks_keep_global_positions(self):
        """Boolean and additive masks preserve the selected global CP rows."""
        for offset in (0, 48):
            allowed = torch.arange(96)[None, :] <= torch.arange(offset, offset + 48)[:, None]
            masks = (allowed, torch.zeros(48, 96).masked_fill(~allowed, float("-inf")))
            for mask in masks:
                with self.subTest(offset=offset, dtype=mask.dtype):
                    arguments = self._invoke(48, 96, mask[None, None])
                    self.assertEqual(arguments["next_tockens"], 1048576)
                    torch.testing.assert_close(arguments["atten_mask"], (~allowed)[None, None])

    def test_explicit_window_overrides_are_preserved(self):
        """The safe default does not override an explicit kernel window choice."""
        owner = torch.nn.Module()
        owner.next_tockens = 17
        self.assertEqual(self._invoke(48, 96, module=owner)["next_tockens"], 17)
        self.assertEqual(self._invoke(48, 96, module=owner, next_tokens=0)["next_tockens"], 0)

    def test_none_causality_uses_module_default(self):
        """Nullable Transformers options retain the ordinary module fallback."""
        owner = torch.nn.Module()
        for default in (True, False):
            with self.subTest(default=default):
                owner.is_causal = default
                expected = self._invoke(48, 96, module=owner)["atten_mask"]
                actual = self._invoke(48, 96, module=owner, is_causal=None)["atten_mask"]
                if default:
                    torch.testing.assert_close(actual, expected)
                else:
                    self.assertIsNone(actual)

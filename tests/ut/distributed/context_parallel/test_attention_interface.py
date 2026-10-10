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
"""Attention-interface CP semantics with only communication isolated."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers.models.llama.modeling_llama import eager_attention_forward
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from hyper_parallel.distributed.context_parallel.attention_interface import _build_attention_interface_cp


class AttentionInterfaceTests(unittest.TestCase):
    """Call the original eager kernel through the real CP mask adapter."""

    def test_call_time_causality_overrides_module_default(self) -> None:
        """Explicit bidirectional/causal calls retain their numerical meaning."""
        mesh = SimpleNamespace(size=lambda: 2, get_local_rank=lambda: 0)
        interface = _build_attention_interface_cp(eager_attention_forward, mesh)
        query = torch.zeros(1, 1, 2, 1)
        keys = torch.zeros(1, 1, 4, 1)
        values = torch.tensor([1., 2., 4., 8.]).reshape(1, 1, 4, 1)
        scenarios = (
            (True, {"is_causal": False}, [3.75, 3.75]),
            (False, {"is_causal": True}, [1., 1.5]),
            (True, {}, [1., 1.5]),
            (False, {}, [3.75, 3.75]),
        )
        for default, options, expected in scenarios:
            with self.subTest(default=default, options=options):
                module = SimpleNamespace(is_causal=default, num_key_value_groups=1, training=False)
                with patch(
                    "hyper_parallel.distributed.context_parallel.attention_interface.flex_cp_allgather",
                    return_value=(keys, values),
                ):
                    actual, _ = interface(module, query, keys[:, :, :2], values[:, :, :2], None,
                                          dropout=0., scaling=1., **options)
                torch.testing.assert_close(actual.flatten(), torch.tensor(expected))

    def test_sdpa_none_causality_uses_module_default(self) -> None:
        """A nullable SDPA option preserves causality at the global query offset."""
        mesh = SimpleNamespace(size=lambda: 2, get_local_rank=lambda: 1)
        interface = _build_attention_interface_cp(sdpa_attention_forward, mesh)
        query = torch.zeros(1, 1, 4, 1)
        keys = torch.zeros_like(query)
        values = torch.tensor([1., 2., 4., 8.]).reshape_as(query)
        for default in (True, False):
            with self.subTest(default=default):
                module = SimpleNamespace(is_causal=default, num_key_value_groups=1, training=False)
                reference, _ = sdpa_attention_forward(module, query, keys, values, None,
                                                       dropout=0., scaling=1., is_causal=None)
                with patch(
                    "hyper_parallel.distributed.context_parallel.attention_interface.flex_cp_allgather",
                    return_value=(keys, values),
                ):
                    actual, _ = interface(module, query[:, :, 2:], keys[:, :, 2:], values[:, :, 2:],
                                          None, dropout=0., scaling=1., is_causal=None)
                torch.testing.assert_close(actual, reference[:, 2:])

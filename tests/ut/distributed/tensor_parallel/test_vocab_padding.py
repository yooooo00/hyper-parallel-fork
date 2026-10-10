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
"""Vocab-parallel embedding preserves native padding output and gradients."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hyper_parallel.distributed._builder.forward_rewriter import _wrap_vocab_parallel_embedding


@pytest.mark.parametrize("padding_idx", [None, 0, 7, -1])
def test_vocab_padding_index_is_local(padding_idx: int | None) -> None:
    """Both owner and non-owner ranks match native output and padding dW."""
    source = nn.Embedding(8, 3, padding_idx=padding_idx)
    with torch.no_grad():
        source.weight.copy_(torch.arange(24, dtype=torch.float32).reshape(8, 3))
    inputs = torch.tensor([[0, 1, 7, 4, 6]])
    expected = source(inputs)
    expected.sum().backward()
    partials = []
    for rank in range(2):
        local = nn.Embedding(8, 3, padding_idx=padding_idx)
        local.weight = nn.Parameter(source.weight.detach().chunk(2)[rank].clone())
        mesh = SimpleNamespace(get_local_rank=lambda index=rank: index)
        _wrap_vocab_parallel_embedding(local, mesh)
        output = local(inputs)
        output.sum().backward()
        torch.testing.assert_close(local.weight.grad, source.weight.grad.chunk(2)[rank], rtol=0, atol=0)
        partials.append(output)
    torch.testing.assert_close(sum(partials), expected, rtol=0, atol=0)

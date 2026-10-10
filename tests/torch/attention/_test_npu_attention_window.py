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
"""Real NPU regression for a local query shard and a global causal mask."""

import json

import pytest
import torch

from hyper_parallel.components.functional import npu_fusion_attention_forward
from tests.torch.attention._test_mxfp8_gqa_attention import _device


@pytest.mark.parametrize("offset", [0, 48])
def test_rectangular_attention_window(offset: int) -> None:
    """Default attention must equal an explicitly unbounded real kernel exactly.

    Args:
        offset: Global starting position of the local query shard.
    """
    device, _ = _device()
    torch.manual_seed(793)
    torch.npu.manual_seed(793)
    query = torch.randn(1, 4, 48, 32, device=device, dtype=torch.bfloat16) * .2
    key = torch.randn(1, 2, 96, 32, device=device, dtype=torch.bfloat16) * .2
    value = torch.randn_like(key)
    gradient = torch.randn(1, 48, 4, 32, device=device, dtype=torch.bfloat16)
    allowed = torch.arange(96, device=device)[None, :] <= torch.arange(offset, offset + 48, device=device)[:, None]
    mask = allowed[None, None]
    owner = torch.nn.Module()
    results = []
    for options in ({}, {"next_tokens": 1048576}):
        states = [tensor.detach().clone().requires_grad_() for tensor in (query, key, value)]
        output, _ = npu_fusion_attention_forward(owner, *states, mask, **options)
        gradients = torch.autograd.grad((output * gradient).sum(), states)
        results.append((output, *gradients))
    for name, actual, expected in zip(("output", "dQ", "dK", "dV"), *results):
        assert torch.isfinite(actual).all(), name
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.npu.synchronize()
    print(json.dumps({"offset": offset, "shape": [1, 4, 48, 96, 32],
                      "default_vs_explicit_window": "exact output/dQ/dK/dV"}), flush=True)

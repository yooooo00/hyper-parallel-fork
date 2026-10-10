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
"""Planner-injected Context Parallel wrappers for Qwen3.5 Gated DeltaNet."""
# pylint: disable=forbidden-backend-import

__all__ = [
    "qwen3_5_gdn_p2p_cp_wrapper",
    "qwen3_5_gdn_ulysses_cp_wrapper",
]

from typing import Any, Type

from torch import nn

from hyper_parallel.distributed import inner_wrapper
from hyper_parallel.distributed.attention_family.state import build_state_cp_rewrite
from hyper_parallel.distributed._builder.forward_rewriter import (
    _ForwardRewriteRequest,
)
from hyper_parallel.distributed.context_parallel.gated_delta_net import (
    GatedDeltaNetP2PCP,
    GatedDeltaNetUlyssesCP,
)


def _build_gdn_cp_rewrite(
    target_module: nn.Module,
    cp_mesh: Any,
    tp_mesh: Any,
    *,
    mode: str,
    executor_class: Type[nn.Module],
    backend: str,
    chunk_size: int,
) -> _ForwardRewriteRequest:
    """Use the shared Family rewrite while preserving model recipe imports."""
    return build_state_cp_rewrite(
        target_module, cp_mesh, tp_mesh, family="gdn", mode=mode,
        executor_class=executor_class, backend=backend, chunk_size=chunk_size,
    )


@inner_wrapper
def qwen3_5_gdn_ulysses_cp_wrapper(
    target_module: nn.Module,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
    backend: str = "eager",
    chunk_size: int = 64,
) -> _ForwardRewriteRequest:
    """Install full-layer Ulysses CP on a Qwen3.5 GDN module."""
    del mesh, ep_mesh
    return _build_gdn_cp_rewrite(
        target_module,
        cp_mesh,
        tp_mesh,
        mode="ulysses",
        executor_class=GatedDeltaNetUlyssesCP,
        backend=backend,
        chunk_size=chunk_size,
    )


@inner_wrapper
def qwen3_5_gdn_p2p_cp_wrapper(
    target_module: nn.Module,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
    backend: str = "eager",
    chunk_size: int = 64,
) -> _ForwardRewriteRequest:
    """Install full-layer recurrent-state P2P CP on a Qwen3.5 GDN module."""
    del mesh, ep_mesh
    return _build_gdn_cp_rewrite(
        target_module,
        cp_mesh,
        tp_mesh,
        mode="p2p",
        executor_class=GatedDeltaNetP2PCP,
        backend=backend,
        chunk_size=chunk_size,
    )

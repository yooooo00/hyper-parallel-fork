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
"""State-aware Family CP adapters with causal convolution halos and state gradients.

The default uses the existing affine-summary P2P executors. Each rank owns a
contiguous, equally sized sequence interval; summaries carry the incoming
state across interval boundaries, and autograd communicates its adjoint back.
Parameters and projection child hooks remain owned by the original module.
"""

from functools import wraps
from inspect import signature
from typing import Any

from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed._builder.forward_rewriter import _ForwardRewriteRequest
from hyper_parallel.distributed.context_parallel.gated_delta_net import GatedDeltaNetP2PCP
from hyper_parallel.distributed.context_parallel.input_validation import reject_packed_sequence_metadata
from hyper_parallel.distributed.context_parallel.kimi_delta_attention import KimiDeltaAttentionLayerP2PCP
from hyper_parallel.distributed.recipe_spec import inner_wrapper


def build_state_cp_rewrite(
    target_module: nn.Module,
    cp_mesh: Any,
    tp_mesh: Any,
    *,
    family: str,
    mode: str,
    executor_class: type[nn.Module],
    backend: str,
    chunk_size: int,
) -> _ForwardRewriteRequest:
    """Construct an atomic state-protocol rewrite shared with explicit recipes.

    Args:
        target_module: Existing state-attention layer retaining all parameters.
        cp_mesh: Sequence-sharding mesh.
        tp_mesh: Optional tensor-parallel mesh; composition is not supported.
        family: Protocol marker, either ``gdn`` or ``kda``.
        mode: Communication algorithm recorded for plan inspection.
        executor_class: Full-layer state-aware CP executor.
        backend: Explicit local arithmetic backend.
        chunk_size: Recurrent scan block length.

    Returns:
        Rewrite request committed by the existing forward rewriter.
    """
    if family not in ("gdn", "kda"):
        raise ValueError(f"Unknown state attention protocol: {family!r}")
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError(f"{family.upper()} {mode} Context Parallel requires an active CP mesh")
    if tp_mesh is not None and tp_mesh.size() > 1:
        raise NotImplementedError(f"{family.upper()} Context Parallel does not support simultaneous TP and CP")
    marker = f"_hp_{family}_cp_config"
    if getattr(target_module, marker, None) is not None:
        raise RuntimeError(f"{family.upper()} Context Parallel has already been applied to this module")
    executor = executor_class(target_module, cp_mesh, backend=backend, chunk_size=chunk_size)
    original_forward = target_module.forward
    forward_signature = signature(original_forward)

    @wraps(original_forward)
    def cp_forward(*args: Any, **kwargs: Any) -> Any:
        """Execute the selected scan without replacing projection modules.

        Args:
            *args: Positional inputs accepted by the original state-attention layer.
            **kwargs: Original keyword inputs, with unsupported cache/packed metadata rejected.

        Returns:
            The full-layer state executor's result in the original forward contract.
        """
        bound = forward_signature.bind_partial(*args, **kwargs)
        options = {**kwargs, **bound.arguments}
        if options.get("use_cache") or any(
            options.get(name) is not None for name in ("cache_params", "past_key_values", "past_key_value")
        ):
            raise ValueError(f"{family.upper()} Family CP supports training without cache only")
        reject_packed_sequence_metadata(options, prefix=f"{family.upper()} Family CP")
        return executor(*args, **kwargs)

    return _ForwardRewriteRequest(
        target_module,
        cp_forward,
        companion_attrs={marker: {"mode": mode, "backend": backend, "chunk_size": chunk_size}},
    )


@inner_wrapper
def gated_delta_net_cp_wrapper(
    target_module: nn.Module,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
    backend: str = "eager",
    chunk_size: int = 64,
) -> _ForwardRewriteRequest:
    """Adapt the exact Qwen3.5 Gated DeltaNet protocol using recurrent state P2P.

    Args:
        target_module: Existing Qwen3.5 Gated DeltaNet layer retaining its parameters.
        mesh: Full runtime mesh, unused by this state protocol.
        tp_mesh: Tensor-parallel submesh; simultaneous active TP is rejected.
        cp_mesh: Active context-parallel mesh for equal contiguous sequence shards.
        ep_mesh: Expert submesh, unused by this state protocol.
        backend: Local scan backend passed to the full-layer executor.
        chunk_size: Local recurrent scan block length.

    Returns:
        Transactional rewrite carrying convolution halos and differentiable recurrent state.
    """
    del mesh, ep_mesh
    return build_state_cp_rewrite(
        target_module, cp_mesh, tp_mesh, family="gdn", mode="p2p",
        executor_class=GatedDeltaNetP2PCP, backend=backend, chunk_size=chunk_size,
    )


@inner_wrapper
def kimi_delta_attention_cp_wrapper(
    target_module: nn.Module,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
    backend: str = "eager",
    chunk_size: int = 64,
) -> _ForwardRewriteRequest:
    """Adapt Kimi Delta Attention using convolution halos and recurrent state P2P.

    Args:
        target_module: Existing Kimi Delta Attention layer retaining its parameters.
        mesh: Full runtime mesh, unused by this state protocol.
        tp_mesh: Tensor-parallel submesh; simultaneous active TP is rejected.
        cp_mesh: Active context-parallel mesh for equal contiguous sequence shards.
        ep_mesh: Expert submesh, unused by this state protocol.
        backend: Local scan backend passed to the full-layer executor.
        chunk_size: Local recurrent scan block length.

    Returns:
        Transactional rewrite carrying convolution halos and differentiable recurrent state.
    """
    del mesh, ep_mesh
    return build_state_cp_rewrite(
        target_module, cp_mesh, tp_mesh, family="kda", mode="p2p",
        executor_class=KimiDeltaAttentionLayerP2PCP, backend=backend, chunk_size=chunk_size,
    )

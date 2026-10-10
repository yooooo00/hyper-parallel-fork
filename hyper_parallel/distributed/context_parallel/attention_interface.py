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
"""Context parallelism for the standard high-performance attention interface.

Protocol: BF16/FP16/FP32 B,H,S,D Q/K/V, contiguous equal sequence shards in
CP-rank order and zero dropout. Q stays local; K/V gather with summed
backward gradients. Masks are boolean allowed masks or additive masks with
zero allowed entries and negative blocked entries, in global coordinates.
The original interface returns B,S,H,D. Cache, sliding windows and packed
sequence metadata require an explicit model adapter.
"""

from __future__ import annotations

from functools import wraps
from inspect import Parameter, signature
from typing import Any, Callable

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed._builder.forward_rewriter import _ForwardRewriteRequest
from hyper_parallel.distributed.context_parallel.collectives import flex_cp_allgather
from hyper_parallel.distributed.context_parallel.input_validation import reject_packed_sequence_metadata
from hyper_parallel.distributed.recipe_spec import inner_wrapper


def _slice_cp_mask(mask: torch.Tensor, query: torch.Tensor, global_length: int, offset: int) -> torch.Tensor:
    """Select this rank's rows from a supported global padding or custom mask."""
    length = query.shape[2]
    if mask.ndim == 2:
        if mask.shape[0] == query.shape[0] == global_length:
            raise ValueError("Family CP ambiguous 2D mask; use explicit [B,1,Q,K] layout")
        if mask.shape[0] == query.shape[0]:
            mask = mask[:, None, None, :]
        elif mask.shape[0] == global_length:
            mask = mask.narrow(0, offset, length)
        else:
            raise ValueError("Family CP 2D mask must be [batch,global_S] or [global_S,global_S]")
    elif mask.ndim == 4:
        if mask.shape[0] not in (1, query.shape[0]) or mask.shape[1] not in (1, query.shape[1]):
            raise ValueError("Family CP mask batch/head dimensions must broadcast to Q")
        if mask.shape[-2] == global_length:
            mask = mask.narrow(-2, offset, length)
        elif mask.shape[-2] != 1:
            raise ValueError("Family CP custom mask must cover all global query rows")
    else:
        raise ValueError("Family CP attention_mask must have 2 or 4 dimensions")
    return mask


def _cp_mask(mask: torch.Tensor | None, query: torch.Tensor, global_length: int,
             offset: int, causal: bool) -> torch.Tensor:
    """Combine global padding/custom masks with this rank's causal row slice."""
    length = query.shape[2]
    rows = torch.arange(length, device=query.device)[:, None] + offset
    columns = torch.arange(global_length, device=query.device)[None, :]
    allowed = (columns <= rows) if causal else torch.ones_like(columns <= rows)
    if mask is None:
        return allowed
    if mask.ndim not in (2, 4):
        raise ValueError("Family CP attention_mask must have 2 or 4 dimensions")
    if mask.dtype != torch.bool and not mask.is_floating_point():
        raise ValueError("Family CP attention_mask must be boolean allowed or floating additive")
    if mask.shape[-1] != global_length:
        raise ValueError("Family CP attention_mask must cover the global key sequence")
    mask = _slice_cp_mask(mask, query, global_length, offset)
    if mask.dtype == torch.bool:
        return allowed & mask
    return mask.masked_fill(~allowed, float("-inf"))


def _validate_states(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    """Reject layouts outside the interface contract before communication."""
    if any(state.ndim != 4 for state in (query, key, value)):
        raise ValueError("Family CP requires B,H,S,D query, key and value tensors")
    if query.shape[0] != key.shape[0] or key.shape[:3] != value.shape[:3]:
        raise ValueError("Family CP requires matching K/V batch, head and sequence dimensions")
    if query.shape[2] != key.shape[2]:
        raise ValueError("Family CP requires equal local Q/K/V sequence lengths; cache is unsupported")
    if query.dtype != key.dtype or key.dtype != value.dtype:
        raise ValueError("Family CP requires matching Q/K/V dtypes")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("Family CP requires matching Q/K head dimensions")
    if len({state.device for state in (query, key, value)}) != 1:
        raise ValueError("Family CP requires Q/K/V on the same device")
    if query.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Family CP requires BF16, FP16 or FP32 tensors")
    if not key.shape[1] or not query.shape[1] or query.shape[1] % key.shape[1]:
        raise ValueError("Family CP query heads must be divisible by KV heads")


def _build_attention_interface_cp(original: Callable, cp_mesh: Any) -> Callable:
    """Preserve the selected kernel while adapting global K/V and mask semantics."""
    @wraps(original)
    def cp_attention_interface(module: Any, query_states: torch.Tensor, key_states: torch.Tensor,
                               value_states: torch.Tensor, attention_mask: torch.Tensor | None,
                               dropout: float = 0.0, scaling: float | None = None,
                               **kwargs: Any) -> tuple[torch.Tensor, Any]:
        """Call the selected kernel with global keys, values and causal positions.

        Args:
            module: Attention module providing the default causal convention.
            query_states: Local B,H,S,D query tensor.
            key_states: Local B,H,S,D key tensor, gathered across CP ranks.
            value_states: Local B,H,S,D value tensor, gathered across CP ranks.
            attention_mask: Global boolean allowed mask or supported additive mask.
            dropout: Attention dropout probability, required to be zero.
            scaling: Optional attention score scaling passed to the original kernel.
            **kwargs: Original kernel options, with unsupported packed/sliding metadata rejected.

        Returns:
            The original kernel's local-query output and auxiliary attention result.
        """
        if dropout != 0.0:
            raise ValueError("Family CP requires zero attention dropout; use a model RNG-aware adapter")
        reject_packed_sequence_metadata(kwargs)
        if kwargs.get("sliding_window") is not None:
            raise ValueError("Family CP does not support sliding windows")
        _validate_states(query_states, key_states, value_states)
        local_length = query_states.shape[2]
        offset = cp_mesh.get_local_rank() * local_length
        is_causal = kwargs.get("is_causal")
        if is_causal is None:
            is_causal = getattr(module, "is_causal", True)
        mask = _cp_mask(attention_mask, query_states, local_length * cp_mesh.size(),
                        offset, is_causal)
        # HF eager uses additive masks whereas NPU attention accepts either
        # convention. Normalize here so the selected kernel keeps its semantics.
        if mask.dtype == torch.bool:
            mask = torch.zeros_like(mask, dtype=query_states.dtype).masked_fill(~mask, float("-inf"))
        keys, values = flex_cp_allgather(key_states.contiguous(), value_states.contiguous(), 2, cp_mesh)
        return original(module, query_states, keys, values, mask,
                        dropout=dropout, scaling=scaling, **kwargs)

    return cp_attention_interface


@inner_wrapper
def attention_interface_cp_wrapper(target_module: Any, mesh: Any, tp_mesh: Any,
                                   cp_mesh: Any, ep_mesh: Any) -> Any:
    """Install CP at an attention interface, preserving projections, RoPE and hooks.

    Args:
        target_module: Attention with the B,H,S,D interface protocol.
        mesh: Full runtime mesh.
        tp_mesh: Dense tensor parallel submesh.
        cp_mesh: Active context parallel submesh.
        ep_mesh: Expert submesh (unused by attention).

    Returns:
        A transactional rewrite request, or None when already installed on this mesh.
    """
    del mesh, tp_mesh, ep_mesh
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError("attention_interface_cp_wrapper requires an active CP mesh")
    installed = getattr(target_module, "_hp_attention_cp_mesh", None)
    if installed is not None:
        if installed is not cp_mesh:
            raise ValueError("Family CP adapter is already bound to a different mesh")
        return None
    original_interface = getattr(target_module, "attention_interface", None)
    if not callable(original_interface):
        raise ValueError("Family CP requires a callable attention_interface")
    if getattr(target_module, "sliding_window", None) is not None:
        raise ValueError("Family CP sliding window attention requires an explicit adapter")
    if getattr(target_module, "attention_dropout", 0.0) != 0.0:
        raise ValueError("Family CP attention dropout requires an explicit RNG-aware adapter")
    original_forward = target_module.forward
    forward_signature = signature(original_forward)
    keyword_parameter = next((name for name, parameter in forward_signature.parameters.items()
                              if parameter.kind == Parameter.VAR_KEYWORD), None)

    @wraps(original_forward)
    def cp_forward(*args: Any, **kwargs: Any) -> Any:
        """Validate sequence-shard and cache contracts before the original forward.

        Args:
            *args: Positional arguments accepted by the original attention forward.
            **kwargs: Original keyword arguments, including any kernel options.

        Returns:
            The original forward result after the CP interface performs communication.
        """
        bound = forward_signature.bind_partial(*args, **kwargs)
        if bound.arguments.get("past_key_values") is not None:
            raise ValueError("Family CP cache is unsupported; use an explicit cache-aware adapter")
        reject_packed_sequence_metadata({**bound.arguments, **bound.arguments.get(keyword_parameter, {})})
        hidden_states = bound.arguments.get("hidden_states")
        if hidden_states is not None:
            if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
                raise ValueError("Family CP hidden_states must have B,S,H layout with 3 dimensions")
            # Every rank participates even for an empty shard, before the base
            # module's empty fast path can skip K/V collectives. This metadata
            # check is required for deterministic failure on mismatched shards.
            shape = tuple(hidden_states.shape[:2])
            shapes = [None] * cp_mesh.size()
            dist.all_gather_object(shapes, shape, group=cp_mesh.get_group())
            if any(other != shape for other in shapes) or not all(shape):
                raise ValueError("Family CP requires equal nonempty batch/sequence shards on every rank")
        return original_forward(*args, **kwargs)

    return _ForwardRewriteRequest(target_module, cp_forward, companion_attrs={
        "attention_interface": _build_attention_interface_cp(original_interface, cp_mesh),
        "_hp_attention_cp_mesh": cp_mesh,
        "_hp_attention_cp_original_interface": original_interface,
    })

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
"""MLA Family CP adapter at the expanded attention-interface boundary.

Latent Q/KV projection, per-component RoPE, and KV expansion stay in the MLA
module. The common interface adapter communicates expanded BNSD keys/values
and sums remote consumers' gradients back into the local compressed latents.
QK and value head widths may differ. This first adapter trades communication
volume for reuse of the model-selected kernel; it does not claim compressed
KV communication or support inference cache.
"""

from typing import Any, Optional

from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed._builder.forward_rewriter import _ForwardRewriteRequest
from hyper_parallel.distributed.context_parallel.attention_interface import attention_interface_cp_wrapper
from hyper_parallel.distributed.recipe_spec import inner_wrapper


@inner_wrapper
def mla_attention_cp_wrapper(
    target_module: nn.Module,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
) -> Optional[_ForwardRewriteRequest]:
    """Adapt fixed-length causal MLA while preserving projections and RoPE.

    Args:
        target_module: MLA module exposing compressed projections and an attention interface.
        mesh: Full runtime mesh, forwarded to the common interface adapter.
        tp_mesh: Tensor-parallel submesh, forwarded without changing TP behavior.
        cp_mesh: Active mesh for contiguous context-parallel sequence shards.
        ep_mesh: Expert submesh, unused by attention.

    Returns:
        Transactional rewrite preserving the selected kernel, or None if already installed on this mesh.

    Raises:
        ValueError: The target lacks the required MLA protocol attributes.
    """
    required = ("q_lora_rank", "kv_lora_rank", "qk_rope_head_dim", "v_head_dim", "attention_interface")
    missing = [name for name in required if not hasattr(target_module, name)]
    if missing:
        raise ValueError(f"MLA attention-interface CP requires the MLA protocol; missing: {missing}")
    return attention_interface_cp_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh)

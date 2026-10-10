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
"""Native Transformers attention fixtures and real Family replacement entry."""

from typing import Any

import torch
from torch import nn
from transformers import LlamaConfig, Qwen3MoeConfig
from transformers.models.llama.modeling_llama import LlamaAttention, eager_attention_forward
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention

from hyper_parallel.models.replacement import _apply_module_replacement_actions
from hyper_parallel.trainer.config import entries_to_module_replacements
from hyper_parallel.trainer.config.resolver import resolve_config


MODEL_TYPES = {"llama": (LlamaConfig, LlamaAttention), "qwen3_moe": (Qwen3MoeConfig, Qwen3MoeAttention)}


def native_attention(model_type: str, *, device: str = "cpu", dtype: torch.dtype = torch.bfloat16,
                     bias: bool = False) -> nn.Module:
    """Create a native GQA with four query heads and two KV heads.

    Args:
        model_type: One of llama or qwen3_moe.
        device: Initial parameter device, including meta.
        dtype: Source projection precision.
        bias: Enable the model's standard projection biases.
    """
    config_class, attention_class = MODEL_TYPES[model_type]
    config = config_class.from_dict({"hidden_size": 128, "num_attention_heads": 4,
                                     "num_key_value_heads": 2, "head_dim": 32,
                                     "attention_bias": bias, "attention_dropout": 0.0,
                                     "num_hidden_layers": 1, "intermediate_size": 256})
    config._attn_implementation = "eager"
    with torch.device(device):
        return attention_class(config, layer_idx=0).to(dtype=dtype)


def replace_attention(source: nn.Module, *, use_mxfp8: bool = False, **arguments: Any) -> tuple[nn.Module, list]:
    """Execute the real parser/replacement without a helper weight reload.

    Args:
        source: Already constructed native attention.
        use_mxfp8: Family selector consumed by production configuration logic.
        **arguments: Other replacement constructor arguments.

    Returns:
        Final attention module and the executor's checkpoint mapping.
    """
    holder = nn.Module()
    holder.attention = source
    source_path = f"{type(source).__module__}.{type(source).__qualname__}"
    config = resolve_config({
        "model": {"_target_": "torch.nn.Identity"},
        "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.01},
        "plan_overrides": [{"match": "attention", "module_type": source_path, "replace_module": {
            "_target_": "hyper_parallel.components.modules.GQAAttention", "use_mxfp8": use_mxfp8,
            "attention_interface": eager_attention_forward, **arguments,
        }}],
    })
    mapping = []
    _apply_module_replacement_actions(holder, entries_to_module_replacements(config.plan_overrides),
                                      weights_mapping=mapping)
    return holder.attention, mapping


def attention_inputs(sequence: int = 32, *, device: str = "cpu", dtype: torch.dtype = torch.bfloat16) -> dict:
    """Create causal attention inputs with nontrivial rotary positions.

    Args:
        sequence: Number of tokens.
        device: Tensor device.
        dtype: Activation and rotary precision.
    """
    hidden = torch.randn(2, sequence, 128, device=device, dtype=dtype) * 0.25
    positions = torch.arange(sequence, device=device, dtype=torch.float32).view(1, sequence, 1)
    frequencies = torch.arange(16, device=device, dtype=torch.float32).view(1, 1, 16) / 100
    angles = (positions * frequencies).repeat(2, 1, 2)
    mask = torch.full((sequence, sequence), torch.finfo(dtype).min, device=device, dtype=dtype).triu(1)
    return {"hidden_states": hidden, "position_embeddings": (angles.cos().to(dtype), angles.sin().to(dtype)),
            "attention_mask": mask[None, None]}

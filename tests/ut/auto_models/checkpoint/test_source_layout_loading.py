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
"""Pretrained loading before HSDP installs its final parameter layouts."""

from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

from hyper_parallel import DTensor, Shard, distribute_tensor
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.mesh import MeshContext
from hyper_parallel.models._transformers.checkpoint_loader import _copy_into_target
from hyper_parallel.models._transformers.model_builder import apply_model_infrastructure


def test_meta_model_loads_plain_tp_shards(tmp_path: Path) -> None:
    """Real meta/planner/apply/load retains exact checkpoint slices after to_empty."""
    config = LlamaConfig.from_dict({"hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 1,
                                    "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 64})
    source = LlamaForCausalLM(config)
    checkpoint = {name: value.detach().clone() for name, value in source.state_dict().items()}
    save_file(checkpoint, tmp_path / "model.safetensors")
    with torch.device("meta"):
        model = LlamaForCausalLM(config)
    mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",), rank_list=(0, 1), init_backend=False)
    context = MeshContext(device_mesh=mesh, fsdp_non_moe_mesh=mesh, tp_size=2)
    with patch("hyper_parallel.distributed._builder.applier.create_tp_collective_lowerer", return_value=None):
        model = apply_model_infrastructure(
            model, mesh=context, sharding_planner=ShardingPlanner(), is_hf_model=True,
            is_meta_device=True, device=torch.device("cpu"), load_base_model=True, pretrained_path=str(tmp_path),
        )
    expected_shards = {"lm_head.weight": 0, "model.layers.0.self_attn.q_proj.weight": 0,
                       "model.layers.0.self_attn.o_proj.weight": 1}
    rank = mesh.get_local_rank()
    for name, dimension in expected_shards.items():
        parameter = model.get_parameter(name)
        assert not isinstance(parameter, DTensor)
        torch.testing.assert_close(parameter, checkpoint[name].chunk(2, dim=dimension)[rank], rtol=0, atol=0)
        assert parameter._is_hf_initialized
    torch.testing.assert_close(model.model.norm.weight, checkpoint["model.norm.weight"], rtol=0, atol=0)
    assert "lm_head.weight" in model._hp_checkpoint_source_shard_info


def test_final_dtensor_layout_takes_priority() -> None:
    """An existing final layout wins over source metadata, avoiding double sharding."""
    mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",), rank_list=(0, 1), init_backend=False)
    full = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    target = torch.nn.Parameter(distribute_tensor(torch.zeros_like(full), mesh, (Shard(1),)))
    _copy_into_target("projection.weight", full, target, ((Shard(0),), mesh))
    torch.testing.assert_close(target.to_local(), full.chunk(2, dim=1)[mesh.get_local_rank()], rtol=0, atol=0)

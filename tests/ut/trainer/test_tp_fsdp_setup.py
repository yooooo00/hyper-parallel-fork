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
"""Trainer must manage TP gradients even when its data mesh has size one."""

from unittest.mock import patch

import pytest

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.distributed.mesh import MeshContext
from hyper_parallel.trainer.config.resolver import resolve_config
from hyper_parallel.trainer.runtime.distributed import create_distributed_setup_from_config


@pytest.mark.parametrize("world_size,tp_size,enabled", [(1, 1, False), (2, 1, True), (2, 2, True)])
def test_tp_only_enables_source_gradient_management(world_size: int, tp_size: int, enabled: bool) -> None:
    """Only process-group construction is isolated; strategy selection stays real."""
    config = resolve_config({"model": {"_target_": "torch.nn.Identity"},
                             "optimizer": {"_target_": "torch.optim.SGD", "lr": .1},
                             "accelerator": {"tp_size": tp_size}})
    dp_size = world_size // tp_size
    mesh = init_device_mesh("cpu", (dp_size, 1, tp_size), mesh_dim_names=("dp", "cp", "tp"),
                            rank_list=tuple(range(world_size)), init_backend=False)
    context = MeshContext(device_mesh=mesh, dp_size=dp_size, dp_replicate_size=dp_size, tp_size=tp_size)
    module = "hyper_parallel.trainer.runtime.distributed"
    with patch(f"{module}.dist.is_initialized", return_value=True), \
            patch(f"{module}.dist.get_world_size", return_value=world_size), \
            patch(f"{module}._build_device_mesh_from_accelerator", return_value=(context, None)):
        setup = create_distributed_setup_from_config(config)
    assert (setup.strategy_config is config.fsdp_config) is enabled
    if not enabled:
        assert setup.strategy_config is None

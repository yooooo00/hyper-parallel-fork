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
"""Regression coverage for materialized HSDP communication storage."""

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.fully_shard.hsdp_param import HSDPParamV2
from hyper_parallel.core.fully_shard.hsdp_utils import ParamModuleInfo
from hyper_parallel.core.fully_shard.utils import FSDPMeshInfo, MixedPrecisionPolicy
from tests.common.mark_utils import arg_mark


class TestMaterializedStorage(unittest.TestCase):
    """Exercise real parameter materialization and autograd with no device collectives."""

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
              card_mark="onecard", essential_mark="essential")
    def test_synchronized_materialized_storage_is_off_graph(self) -> None:
        """
        Feature: HSDP storage after meta materialization.
        Description: Synchronize live weights before unsharding, then repeat a gradient/update cycle.
        Expectation: Storage aliases remain live without an autograd view error or stale weights.
        """
        for trainable in (False, True):
            with self.subTest(trainable=trainable):
                module = torch.nn.Linear(4, 4, bias=False, device="meta")
                mesh_info = object.__new__(FSDPMeshInfo)
                with patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", return_value=0):
                    mesh_info.mesh = DeviceMesh(
                        "cpu", [0], mesh_dim_names=("fsdp",), _init_backend=False,
                    )
                mesh_info.shard_mesh_dim = 0
                mesh_info.replicate_mesh_dim = None
                mesh_info.shard_mesh_rank = 0
                mesh_info.shard_mesh_size = 1
                mesh_info.shard_process_group = None
                parameter = HSDPParamV2(
                    module.weight, ParamModuleInfo(module, "weight", [], []), mesh_info,
                    mp_policy=MixedPrecisionPolicy(), device=torch.device("cpu"),
                )
                sharded_parameter = parameter.sharded_param
                sharded_parameter._local_tensor = torch.ones(4, 4, requires_grad=trainable)
                with torch.no_grad():
                    parameter.reset_sharded_param()
                # Parameter synchronization writes storage outside the forward graph.
                sharded_parameter._local_tensor.detach().add_(1)
                parameter.unshard()
                parameter.wait_for_unshard()
                live_parameter = parameter.unsharded_param
                self.assertIs(parameter.sharded_param, sharded_parameter)
                self.assertFalse(parameter._sharded_param_data.requires_grad)
                self.assertIsNone(parameter._sharded_param_data.grad_fn)
                self.assertEqual(parameter._sharded_param_data.data_ptr(), live_parameter.data_ptr())
                torch.testing.assert_close(live_parameter, torch.full((4, 4), 2.0))
                if trainable:
                    live_parameter.square().sum().backward()
                    torch.testing.assert_close(live_parameter.grad, torch.full((4, 4), 4.0))
                parameter.to_sharded()
                sharded_parameter._local_tensor.detach().add_(1)
                parameter.unshard()
                parameter.wait_for_unshard()
                self.assertIs(parameter.unsharded_param, live_parameter)
                torch.testing.assert_close(live_parameter, torch.full((4, 4), 3.0))

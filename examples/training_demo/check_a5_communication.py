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
"""Check A5 HCCL initialization independently of HyperParallel.

Run with ``torchrun --standalone --nproc-per-node=2``. Select idle physical
cards with ASCEND_RT_VISIBLE_DEVICES before launching the process.
"""

import os
from datetime import timedelta
from pathlib import Path

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import
import torch_npu


def main() -> None:
    """Validate the root-info path and verify a synchronized HCCL reduction."""
    root_info = Path("/etc/hccl_rootinfo.json")
    if root_info.exists() and not root_info.is_file():
        raise RuntimeError(
            f"{root_info} exists but is not a regular file. HCCL may try to "
            "parse it as JSON and fail during communicator initialization. "
            "Inspect the host configuration before running distributed tests."
        )
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", timeout=timedelta(seconds=300))
    try:
        value = torch.tensor([dist.get_rank() + 1.0], device="npu")
        dist.all_reduce(value)
        actual = value.item()
        world_size = dist.get_world_size()
        expected = world_size * (world_size + 1) / 2
        if actual != expected:
            raise RuntimeError(f"HCCL reduction returned {actual}; expected {expected}.")
        print(
            f"PASS rank={dist.get_rank()} world_size={world_size} sum={actual} "
            f"torch={torch.__version__} torch_npu={torch_npu.__version__}",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

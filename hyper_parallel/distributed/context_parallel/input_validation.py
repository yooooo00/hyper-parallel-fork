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
"""Shared input guards for fixed-length context-parallel adapters."""

from collections.abc import Mapping
from typing import Any


_PACKED_SEQUENCE_ARGUMENTS = (
    "packed_seq_params", "actual_seq_len", "actual_q_len", "actual_kv_len",
    "actual_seq_qlen", "actual_seq_kvlen", "cu_seq_lens", "cu_seq_lens_q",
    "cu_seq_lens_k", "cu_seq_lens_kv", "cu_seqlens", "cu_seqlens_q",
    "cu_seqlens_k", "cu_seqlens_kv", "seq_idx",
)


def reject_packed_sequence_metadata(arguments: Mapping[str, Any], *, prefix: str = "Family CP") -> None:
    """Reject packed aliases before a fixed-length adapter can ignore them.

    Args:
        arguments: Bound forward arguments, including expanded keyword arguments.
        prefix: Attention protocol named in the capability error.
    """
    provided = [name for name in _PACKED_SEQUENCE_ARGUMENTS if arguments.get(name) is not None]
    if provided:
        raise ValueError(
            f"{prefix} packed sequence metadata ({', '.join(provided)}) requires an explicit adapter"
        )

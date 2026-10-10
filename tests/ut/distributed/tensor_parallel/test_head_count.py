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
"""Grouped QKV Family projection integration with the existing TP head plan."""

import unittest

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec, TP
from hyper_parallel.distributed.tensor_parallel.head_count import collect_auto_head_attrs, update_module_head_counts
from tests.common.gqa_family import native_attention, replace_attention


class GroupedQKVHeadCountTests(unittest.TestCase):
    """Local cached head counts follow the final interleaved projection plan."""

    def test_grouped_projection_adjusts_only_cached_heads(self) -> None:
        """TP2 preserves global config metadata and the query/KV grouping ratio."""
        attention, _ = replace_attention(native_attention("llama"))
        spec = ModuleShardingSpec(params={"linear_qkv.weight": {TP: Shard(0)}})
        self.assertEqual(collect_auto_head_attrs(attention, spec, ("tp",)),
                         ("num_heads", "num_key_value_heads"))
        self.assertEqual(update_module_head_counts(attention, 2, "self_attn"), 2)
        self.assertEqual((attention.num_heads, attention.num_key_value_heads), (2, 1))
        self.assertEqual((attention.config.num_attention_heads, attention.config.num_key_value_heads), (4, 2))
        self.assertEqual(attention.num_key_value_groups, 2)
        self.assertEqual(update_module_head_counts(attention, 2, "self_attn"), 0)

    def test_replicated_projection_does_not_request_adjustment(self) -> None:
        """Merely recognizing the name must not divide unsharded head metadata."""
        attention, _ = replace_attention(native_attention("llama"))
        spec = ModuleShardingSpec(params={"linear_qkv.weight": {TP: Replicate()}})
        self.assertEqual(collect_auto_head_attrs(attention, spec, ("tp",)), ())

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
"""DSA metadata contracts isolated from the unavailable optional Omni backend."""

import subprocess
import sys
import textwrap
import unittest


class DSAMetadataTests(unittest.TestCase):
    """No mock sparse arithmetic is used as evidence of device correctness."""

    def test_masks_loaded_weights_and_compatibility_import(self):
        """Isolate only optional backend registration in a disposable interpreter."""
        program = textwrap.dedent("""
            import sys
            from types import ModuleType
            from unittest.mock import patch
            import torch
            from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
            from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Attention
            from hyper_parallel.models.replacement import _apply_module_replacement_actions
            from hyper_parallel.trainer.config import entries_to_module_replacements
            from hyper_parallel.trainer.config.resolver import resolve_config

            # Import registration is isolated; no sparse kernel is executed or mocked.
            with patch.dict(sys.modules, {"omni_training_custom_ops": ModuleType("omni_training_custom_ops")}):
                from hyper_parallel.distributed.attention_family.dsa import (
                    _validate_cp_attention_mask, dsa_attention_cp_wrapper,
                )
                from hyper_parallel.models.deepseek_v32.adapter.distributed.context_parallel import (
                    deepseek_v32_dsa_attention_cp_wrapper,
                )
                assert deepseek_v32_dsa_attention_cp_wrapper is dsa_attention_cp_wrapper
                dims = dict(batch_size=1, local_seq_len=2, global_seq_len=4, cp_rank=1)
                allowed = torch.ones(4, 4, dtype=torch.bool).tril()[None, None]
                _validate_cp_attention_mask(allowed, **dims)
                _validate_cp_attention_mask(allowed[:, :, 2:], **dims)
                additive = torch.zeros_like(allowed, dtype=torch.float32).masked_fill(~allowed, float("-inf"))
                _validate_cp_attention_mask(additive, **dims)
                for invalid in (torch.ones_like(allowed), allowed.clone(), additive.clone()):
                    invalid[..., -1, 0] = False if invalid.dtype == torch.bool else -1.
                    try:
                        _validate_cp_attention_mask(invalid, **dims)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("custom or padded DSA masks must not be ignored")

                config = DeepseekV32Config.from_dict({
                    "hidden_size": 16, "num_attention_heads": 2, "num_hidden_layers": 1,
                    "q_lora_rank": 8, "kv_lora_rank": 8, "qk_rope_head_dim": 4,
                    "qk_nope_head_dim": 4, "v_head_dim": 4,
                    "index_n_heads": 2, "index_head_dim": 8, "index_topk": 4,
                })
                holder = torch.nn.Module()
                holder.attention = DeepseekV32Attention(config, layer_idx=0)
                source = holder.attention
                source.q_a_proj.weight.requires_grad_(False)
                source.kv_a_proj_with_mqa.weight.requires_grad_(False)
                expected = torch.cat((source.q_a_proj.weight, source.kv_a_proj_with_mqa.weight))
                parsed = resolve_config({
                    "model": {"_target_": "torch.nn.Identity"},
                    "optimizer": {"_target_": "torch.optim.SGD", "lr": .01},
                    "plan_overrides": [{"match": "attention",
                        "module_type": "transformers.models.deepseek_v32.modeling_deepseek_v32.DeepseekV32Attention",
                        "replace_module": {"_target_": "hyper_parallel.components.modules.DeepseekV32DSAAttention"}}],
                })
                _apply_module_replacement_actions(
                    holder, entries_to_module_replacements(parsed.plan_overrides), weights_mapping=[],
                )
                torch.testing.assert_close(holder.attention.linear_qkv.weight, expected, atol=0, rtol=0)
                assert not holder.attention.linear_qkv.weight.requires_grad
        """)
        result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

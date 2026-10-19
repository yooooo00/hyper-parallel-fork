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
"""Prepare offline, randomly initialized Qwen3-MoE Trainer acceptance fixtures."""

import argparse
import copy
import json
from pathlib import Path

import torch  # pylint: disable=forbidden-backend-import
import yaml
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3MoeConfig, Qwen3MoeForCausalLM


def prepare(output_dir: Path) -> None:
    """Write native HF weights, synthetic text, tokenizer and six Trainer YAMLs.

    Args:
        output_dir: Local directory containing generated validation assets.
    """
    output_dir = output_dir.resolve()
    assets = output_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    words = (
        "the cat sits on mat a dog runs in park we train small model with "
        "fixed local text and compare expert gradients tokens weights loss"
    ).split()
    vocabulary = {word: index for index, word in enumerate(["[PAD]", "[UNK]", "[EOS]"] + sorted(set(words)))}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]",
    ).save_pretrained(assets)
    config = Qwen3MoeConfig.from_dict({
        "vocab_size": 128, "hidden_size": 128, "intermediate_size": 256, "moe_intermediate_size": 256,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
        "num_experts": 4, "num_experts_per_tok": 2, "max_position_embeddings": 512,
        "use_cache": False, "pad_token_id": 0, "eos_token_id": 2,
    })
    torch.manual_seed(42)
    model = Qwen3MoeForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(assets)
    with (output_dir / "text.jsonl").open("w", encoding="utf-8") as stream:
        for index in range(128):
            offset = index % len(words)
            stream.write(json.dumps({"text": " ".join(words[offset:] + words[:offset]) + " [EOS]"}) + "\n")

    template = Path(__file__).with_name("qwen3_moe") / "train_parallel_online.yaml"
    base = yaml.safe_load(template.read_text(encoding="utf-8"))
    base["model"] = {
        "_target_": "hyper_parallel.models.HyperAutoModelForCausalLM.from_pretrained",
        "pretrained_model_name_or_path": str(assets), "local_files_only": True,
        "torch_dtype": "bfloat16", "attn_implementation": "sdpa", "validate_placement": False,
    }
    base["training"].update(train_iters=10, global_batch_size=2)
    base["accelerator"].update(tp_size=1, cp_size=1, pp_size=1, sequence_parallel=False, loss_parallel=False)
    base["fsdp_config"].update(dp_shard_size=1, edp_shard_size=1)
    base["dataset"]["model_assets"]["tokenizer"]["pretrained_model_name_or_path"] = str(assets)
    base["dataset"]["data_path"] = str(output_dir / "text.jsonl")
    base["dataset"]["data_transform"]["max_seq_len"] = 64
    # Keep one source sample per micro-batch so train_iters counts real updates.
    base["dataloader"]["token_budget"] = 1
    base["optimizer"] = {
        "_target_": "hyper_parallel.components.optim.AdamW",
        "adamw_config": {"adamw_lr": 0.0001, "adamw_weight_decay": 0.01,
                         "adamw_betas": [0.9, 0.999], "adamw_eps": 1e-8},
    }
    base["lr_scheduler"]["lr_config"].update(lr=0.0001, lr_warmup_steps=0)
    for ep_size in (1, 2):
        for mode in ("bf16", "mxfp8", "fused"):
            case = copy.deepcopy(base)
            case["accelerator"]["ep_size"] = ep_size
            case["plan_overrides"] = [
                {"match": "*.mlp.experts",
                 "module_type": "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeExperts",
                 "replace_module": {
                    "_target_": "hyper_parallel.components.modules.GroupedExperts",
                    "use_mxfp8": mode != "bf16", "use_fused_mxfp8": mode == "fused",
                }},
                {"match": "*.mlp", "when": "ep", "region_dispatch": False, "local_compute_fn": {
                    "_target_": (
                        "hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel."
                        "qwen3moe_ep_compute_fn"
                    ),
                }},
            ]
            name = f"ep{ep_size}-{mode}"
            case["checkpoint"].update(
                save_ckpt=True, checkpoint_dir=str(output_dir / f"{name}-ckpt"), save_steps=5, is_async=False,
            )
            (output_dir / f"{name}.yaml").write_text(yaml.safe_dump(case, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    prepare(parser.parse_args().output_dir)

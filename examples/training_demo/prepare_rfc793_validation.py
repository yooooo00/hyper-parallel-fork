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
"""Generate offline Qwen3-MoE Family topology configurations using real text."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
import yaml
from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM

from examples.training_demo.prepare_family_validation import prepare


TOPOLOGIES = (
    (1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2),
    (2, 2, 1), (2, 1, 2), (1, 2, 2), (2, 2, 2),
)
MODES = ("bf16", "mxfp8", "fused")
SEED = 42
SEQUENCE_LENGTH = 128


def _prepare_assets(output_dir: Path, tokenizer_path: Path, data_path: Path) -> Qwen3MoeConfig:
    """Replace the small synthetic fixture with official tokenization and local text."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    prepare(output_dir)
    assets = output_dir / "assets"
    config = Qwen3MoeConfig.from_pretrained(assets, local_files_only=True)
    config.vocab_size = ((len(tokenizer) + 127) // 128) * 128
    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    torch.manual_seed(SEED)
    Qwen3MoeForCausalLM(config).to(torch.bfloat16).save_pretrained(assets)
    tokenizer.save_pretrained(assets)
    (output_dir / "text.jsonl").unlink()
    (output_dir / "text.jsonl").symlink_to(data_path)
    return config


def _replicated_expert_override() -> dict[str, Any]:
    """Keep packed experts replicated when only the attention projections use TP."""
    parameters = ("gate.weight", "experts.gate_up_proj", "experts.down_proj")
    return {
        "match": "*.mlp",
        "params": {name: {"tp": "replicate", "cp": "replicate"} for name in parameters},
        "in_src": {"hidden_states": {"tp": "replicate", "cp": "shard(1)"}},
        "in_dst": {"hidden_states": {"tp": "replicate", "cp": "shard(1)"}},
        "out_src": {"tp": "replicate", "cp": "shard(1)"},
        "out_dst": {"tp": "replicate", "cp": "shard(1)"},
    }


def _build_case(source: dict[str, Any], topology: tuple[int, int, int], mode: str) -> dict[str, Any]:
    """Select Family variants and topology through the production config schema."""
    tp, cp, ep = topology
    case = copy.deepcopy(source)
    case["accelerator"].update(tp_size=tp, cp_size=cp, ep_size=ep)
    case["training"]["train_iters"] = 50 if topology == (2, 2, 2) else 10
    case["plan_overrides"].insert(0, {
        "match": "*.self_attn",
        "module_type": "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeAttention",
        "replace_module": {
            "_target_": "hyper_parallel.components.modules.GQAAttention",
            "use_mxfp8": mode != "bf16",
            "use_fused_mxfp8": False,
        },
    })
    experts = case["plan_overrides"][1]["replace_module"]
    experts.update(use_mxfp8=mode != "bf16", use_fused_mxfp8=mode == "fused")
    if tp > 1 and ep == 1:
        # Packed experts have no internal matrix TP plan. GQA-only TP cases
        # explicitly retain their replicated weights and activations.
        case["plan_overrides"].append(_replicated_expert_override())
    return case


def _write_case(output_dir: Path, name: str, case: dict[str, Any], *, explicit_cp: bool = False) -> None:
    """Write one independent checkpoint destination and optionally select CP explicitly."""
    case = copy.deepcopy(case)
    checkpoint_name = name + "-ckpt"
    if explicit_cp:
        name += "-explicit"
        checkpoint_name += "-explicit"
        case["plan_overrides"].append({
            "match": "*.self_attn", "when": "cp", "inner_target": "self", "region_dispatch": False,
            "inner_wrapper": {"_target_": (
                "hyper_parallel.distributed.context_parallel.attention_interface.attention_interface_cp_wrapper"
            )},
        })
    case["checkpoint"]["checkpoint_dir"] = str(output_dir / checkpoint_name)
    (output_dir / f"{name}.yaml").write_text(yaml.safe_dump(case, sort_keys=False), encoding="utf-8")


def _write_manifest(output_dir: Path, tokenizer_path: Path, data_path: Path, vocab_size: int) -> None:
    """Record the reproducibility inputs without claiming pretrained model weights."""
    manifest = {
        "model": "random-initialized Qwen3MoeForCausalLM", "seed": SEED,
        "tokenizer_path": str(tokenizer_path), "data_path": str(data_path),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256((tokenizer_path / "tokenizer.json").read_bytes()).hexdigest(),
        "vocab_size": vocab_size, "sequence_length": SEQUENCE_LENGTH,
        "preprocessing": "plaintext windows, EOS per source row, continuous positions and causal attention",
    }
    (output_dir / "assets-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def prepare_rfc(output_dir: Path, tokenizer_path: Path, data_path: Path) -> None:
    """Build fixed random weights, real-tokenizer data and complete Trainer YAMLs.

    Args:
        output_dir: New acceptance directory; existing assets are never overwritten.
        tokenizer_path: Local official Qwen tokenizer snapshot (no weight download).
        data_path: Pinned local WikiText train JSONL.

    Raises:
        FileExistsError: The output directory already exists.
        FileNotFoundError: The local dataset is absent.
    """
    output_dir = output_dir.resolve()
    tokenizer_path = tokenizer_path.resolve()
    data_path = data_path.resolve(strict=True)
    if output_dir.exists():
        raise FileExistsError(f"Use a fresh output directory: {output_dir}")
    config = _prepare_assets(output_dir, tokenizer_path, data_path)
    source = yaml.safe_load((output_dir / "ep1-bf16.yaml").read_text(encoding="utf-8"))
    source["dataset"]["data_transform"]["max_seq_len"] = SEQUENCE_LENGTH
    # Continuous positions and a global causal mask define the first CP
    # protocol. Packed-document resets require a model-specific adapter.
    source["dataset"]["data_config"].update(reset_position_ids=False, reset_attention_mask=False)
    for topology in TOPOLOGIES:
        tp, cp, ep = topology
        for mode in MODES:
            name = f"tp{tp}-cp{cp}-ep{ep}-{mode}"
            case = _build_case(source, topology, mode)
            _write_case(output_dir, name, case)
            if cp > 1:
                _write_case(output_dir, name, case, explicit_cp=True)
    _write_manifest(output_dir, tokenizer_path, data_path, config.vocab_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    args = parser.parse_args()
    prepare_rfc(args.output_dir, args.tokenizer_path, args.data_path)

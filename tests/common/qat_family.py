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
"""Shared configuration fixtures for QAT Family integration tests."""

from hyper_parallel.models.replacement import apply_module_replacements, compile_module_replacements
from hyper_parallel.trainer.config.parallelism import entries_to_module_replacements
from hyper_parallel.trainer.config.resolver import resolve_config
from hyper_parallel.trainer.config import TrainerConfig
from tests.common.grouped_experts_family import TinyMoE, family_config


def qat_config(*, fake: bool = False, block_size: int = 32, weight_format: str = "mxfp4") -> TrainerConfig:
    """Resolve a named scheme through the actual training configuration parser.

    Args:
        fake: Select the existing fake W4A8 strategy.
        block_size: Quantization block size.
        weight_format: Weight quantization format.
    """
    raw = family_config(use_low_precision=True).to_dict()
    raw["training"]["low_precision"] = {
        "enabled": True, "dtype_schemes": {"experts": {
            "weight_format": weight_format, "act_format": "mxfp8",
            "is_fake_quantize": fake, "block_size": block_size}},
    }
    raw["plan_overrides"][0].update(when="low_precision", low_precision_dtype_scheme="experts")
    return resolve_config(raw)


def install_qat(model: TinyMoE, config: TrainerConfig) -> TinyMoE:
    """Execute the existing resolver's policy injection and replacement engine.

    Args:
        model: Tiny model with packed experts.
        config: Resolved training configuration.
    """
    rules = entries_to_module_replacements(config.plan_overrides, low_precision_enabled=True,
                                            low_precision_config=config.training.low_precision)
    return apply_module_replacements(model, compile_module_replacements(model, rules), weights_mapping=[])[0]

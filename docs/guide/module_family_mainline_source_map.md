# 第一阶段累计来源映射

来源：`36a9653b..32e2dd96`，包含 d7e06241 本身，共四个提交、31 个文件。

| 原路径 | 迁移路径 | 处置 |
| --- | --- | --- |
| `.codespellrc` | `.codespellrc` | 主线已包含 cann，并保留 tbe/enque；无额外改动 |
| `docs/guide/grouped_experts_mxfp8.md` | `docs/guide/grouped_experts_mxfp8.md` | 保留带原日期的历史设计/验证资料；不作为本轮验收证据 |
| `docs/guide/grouped_experts_mxfp8_a5_validation.md` | `docs/guide/grouped_experts_mxfp8_a5_validation.md` | 保留带原日期的历史设计/验证资料；不作为本轮验收证据 |
| `docs/guide/grouped_experts_mxfp8_fusion.md` | `docs/guide/grouped_experts_mxfp8_fusion.md` | 保留带原日期的历史设计/验证资料；不作为本轮验收证据 |
| `docs/guide/module_family_implementation.md` | `docs/guide/module_family_implementation.md` | 从四提交累计范围迁入；必要时适配新主线接口 |
| `docs/guide/module_family_selector_rules_validation.md` | `docs/guide/module_family_selector_rules_validation.md` | 保留带原日期的历史设计/验证资料；不作为本轮验收证据 |
| `examples/training_demo/prepare_family_validation.py` | `examples/training_demo/prepare_family_validation.py` | 从四提交累计范围迁入；必要时适配新主线接口 |
| `hyper_parallel/components/modules/__init__.py` | `hyper_parallel/components/modules/__init__.py` | 保留第一阶段功能并与当前主线合并 |
| `hyper_parallel/components/modules/family.py` | `hyper_parallel/components/modules/family.py` | 从四提交累计范围迁入；必要时适配新主线接口 |
| `hyper_parallel/components/modules/grouped_experts.py` | `hyper_parallel/components/modules/grouped_experts.py` | 保留第一阶段功能并与当前主线合并 |
| `hyper_parallel/components/modules/mxfp8_grouped_swiglu.py` | `hyper_parallel/components/modules/mxfp8_grouped_swiglu.py` | 从四提交累计范围迁入；必要时适配新主线接口 |
| `hyper_parallel/components/quantization/functional/_saved_quantized.py` | `hyper_parallel/components/quantization/functional/_saved_quantized.py` | da9aca5f：迁入 pack/unpack；公共 GroupedLinear 使用物理保存 |
| `hyper_parallel/components/quantization/functional/mxfp8_fused_grouped_swiglu.py` | `hyper_parallel/components/quantization/functional/mxfp8_fused_grouped_swiglu.py` | 从四提交累计范围迁入；必要时适配新主线接口 |
| `hyper_parallel/components/quantization/functional/npu_mxfp8.py` | `hyper_parallel/components/quantization/ops/npu_mxfp8.py` | 融合能力检查及两个 adapter 迁入 ops 路径；不恢复旧路径 |
| `hyper_parallel/distributed/_builder/module_family.py` | `hyper_parallel/distributed/_builder/module_family.py` | 从四提交累计范围迁入；必要时适配新主线接口 |
| `hyper_parallel/distributed/apply.py` | `hyper_parallel/distributed/apply.py` | 保留第一阶段功能并与当前主线合并 |
| `hyper_parallel/distributed/expert_parallel/experts.py` | `hyper_parallel/distributed/expert_parallel/experts.py` | 保留第一阶段功能并与当前主线合并 |
| `hyper_parallel/distributed/expert_parallel/recipes.py` | `hyper_parallel/distributed/expert_parallel/recipes.py` | 保留第一阶段功能并与当前主线合并 |
| `hyper_parallel/models/qwen3_moe/adapter/distributed/expert_parallel.py` | `hyper_parallel/models/qwen3_moe/adapter/distributed/expert_parallel.py` | 保留第一阶段功能并与当前主线合并 |
| `hyper_parallel/models/replacement.py` | `hyper_parallel/models/replacement.py` | 保留第一阶段功能并与当前主线合并 |
| `hyper_parallel/trainer/config/parallelism.py` | `hyper_parallel/trainer/config/parallelism.py` | 保留第一阶段功能并与当前主线合并 |
| `hyper_parallel/trainer/config/replacement.py` | `hyper_parallel/trainer/config/replacement.py` | 从四提交累计范围迁入；必要时适配新主线接口 |
| `hyper_parallel/trainer/config/resolver.py` | `hyper_parallel/trainer/config/resolver.py` | 保留第一阶段功能并与当前主线合并 |
| `tests/common/grouped_experts_family.py` | `tests/common/grouped_experts_family.py` | 从四提交累计范围迁入；必要时适配新主线接口 |
| `tests/torch/expert_parallel/_test_grouped_experts_mxfp8.py` | `tests/torch/expert_parallel/_test_grouped_experts_mxfp8.py` | 从四提交累计范围迁入；必要时适配新主线接口 |
| `tests/torch/expert_parallel/compare_family_training.py` | `tests/torch/expert_parallel/compare_family_training.py` | 从四提交累计范围迁入；必要时适配新主线接口 |
| `tests/torch/expert_parallel/family_trainer_validation.py` | `tests/torch/expert_parallel/family_trainer_validation.py` | 从四提交累计范围迁入；必要时适配新主线接口 |
| `tests/torch/expert_parallel/test_grouped_experts_mxfp8.py` | `tests/torch/expert_parallel/test_grouped_experts_mxfp8.py` | 从四提交累计范围迁入；必要时适配新主线接口 |
| `tests/ut/components/modules/test_module_family.py` | `tests/ut/components/modules/test_module_family.py` | 从四提交累计范围迁入；必要时适配新主线接口 |
| `tests/ut/components/modules/test_module_family_selectors.py` | `tests/ut/components/modules/test_module_family_selectors.py` | 从四提交累计范围迁入；必要时适配新主线接口 |
| `tests/ut/dual_mode_dtensor/test_ep_moe_local_region.py` | `tests/ut/dual_mode_dtensor/test_ep_moe_local_region.py` | 保留第一阶段功能并与当前主线合并 |

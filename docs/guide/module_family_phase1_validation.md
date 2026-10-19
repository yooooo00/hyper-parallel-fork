# Module Family 第一阶段 PR 验证

日期：2026-10-19。基线为官方主线 `facdbf844fb7546ac133c049b72e76ee5eb93af0`。
本 PR 是 RFC #793 的第一阶段评审版本，包含声明式 rule、GroupedExperts 三种实现及 EP 接入。
没有实现通用 QAT Family 入口，也不宣称完成整个 RFC。

## 结果

以下结果来自本 PR 候选代码在新主线上的重新运行，不引用旧分支通过记录替代：

| 验证 | 结果 |
| --- | --- |
| CPU 相关回归 | 505 passed、752 subtests passed |
| 原 A5 完整入口 | 8 passed、0 failed、0 skipped |
| EP1/EP2 × BF16/MXFP8/融合 Trainer | 6/6 通过，每路实际完成10步 |
| 六路第5步检查点的新进程恢复 | 6/6 通过，第6–10步 loss、梯度范数、权重、样本和 RNG 精确一致 |
| 融合 EP1/EP2 完整非重入重算 | 2/2 通过，第1–10步 loss、梯度范数、权重和 RNG 精确一致 |
| 完整 AutoGit 静态检查 | 未通过，见下文 |

A5 用例包含真实算子、输入/权重梯度、重复反向、空输入、零 token 专家、冻结权重、优化器更新，
以及 planner 分片后的跨 rank 路由、空接收 rank 和融合/非融合对照。
观察器检查实际 MXFP8 GMM 及融合前向/反向调用，避免静默退回 BF16。
数值阈值沿用现有测试。检查点恢复与重算的权重采用 `rtol=0, atol=0`，其它对应字段使用相等检查。
跨精度训练 loss 记录差异，不以事后挑选的阈值作为通过依据。

十步轨迹中跨精度 loss 的最大绝对差如下（仅记录观测值，不作为收敛结论）：

| 对比 | EP=1 | EP=2 |
| --- | --- | --- |
| fused-vs-mxfp8 | 0.00058079 | 0.00135207 |
| fused-vs-bf16 | 0.00079775 | 0.00143933 |
| mxfp8-vs-bf16 | 0.00067663 | 0.00118279 |

## 环境与资产

- 设备：Ascend 950PR（A5），测试使用两张卡。
- Python 3.10.21；torch 2.9.0+cpu；torch_npu 2.9.0.post7+gitddd16dd；Transformers 5.12.0。
- CANN 9.2.0 beta.2；npu-smi/驱动版本25.1.rc1.b087。
- 原生 Qwen3-MoE 缩小配置：2层、H=128、I=256、4专家、top-k=2、vocab=128，BF16 持久参数。
- 模型随机初始化，tokenizer 和固定合成文本由仓库脚本离线生成；不是预训练模型精度或标准数据集收敛验证。

## 复现

在已安装匹配 CANN、torch_npu 和本仓库 editable 包的环境中执行。
`ASCEND_RT_VISIBLE_DEVICES` 应指定空闲物理卡；运行前加载本机 CANN 环境脚本。

```bash
export OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120

python -m pytest -q \
  tests/ut/components/quantization \
  tests/ut/components/modules/test_module_family.py \
  tests/ut/components/modules/test_module_family_selectors.py \
  tests/ut/dual_mode_dtensor/test_ep_moe_local_region.py \
  tests/ut/dual_mode_dtensor/test_module_replacement.py \
  tests/ut/models/deepseek_v3/adapter/test_qat_replacements.py \
  tests/ut/trainer/test_config_overrides.py \
  tests/ut/core/expert_parallel \
  tests/ut/auto_models/models/qwen3_moe/test_replacements.py \
  tests/ut/auto_models/trainer/test_host_optimizer_restore.py \
  tests/ut/trainer/test_optimizer.py \
  tests/ut/core/fully_shard

# 这是 launcher 测试入口，不要在外层再套 torchrun。
ASCEND_RT_VISIBLE_DEVICES=0,1 python -m pytest -q -s \
  tests/torch/expert_parallel/test_grouped_experts_mxfp8.py

# 必须使用新目录，避免混合多次运行的轨迹。
VALIDATION_DIR="$(mktemp -d)"
python -m examples.training_demo.prepare_family_validation "$VALIDATION_DIR"
for ep in 1 2; do
  for mode in bf16 mxfp8 fused; do
    name="ep${ep}-${mode}"
    ASCEND_RT_VISIBLE_DEVICES=0,1 FAMILY_TRACE_DIR="$VALIDATION_DIR/trace-$name" \
      python -m torch.distributed.run --standalone --nproc-per-node="$ep" \
      -m tests.torch.expert_parallel.family_trainer_validation "$VALIDATION_DIR/$name.yaml"
  done
done
python -m tests.torch.expert_parallel.compare_family_training "$VALIDATION_DIR"
```

恢复验证在相应配置中设置 `checkpoint.restore_from=<对应 checkpoint_dir>/global_step_5`，
并设置 `checkpoint.save_ckpt=false`；保持总步数10，使用新进程和 `resume-epN-mode` 轨迹目录。
完整重算使用 `activation_checkpoint.mode=full`，轨迹目录为 `ac-epN-fused`。
全部运行后使用比较器的 `--resume --activation-checkpoint` 执行逐项检查。

## 依赖修复与评审边界

公共 GroupedLinear 保存物理 Tensor、在反向重建临时包装，以支持重复反向、版本检查和 saved-tensor hooks。
HSDP 的物化存储修复对通信视图执行 detach，避免同步后首次 unshard 的 autograd 视图错误；
该缺陷曾在未加入 Family 的纯主线上复现。两个修复均有独立回归，可按维护者意见拆分。

完整静态检查当前仍报告已有文件的文档、类型标注、测试标记/设计描述和复杂度问题，
以及与主线 Torch-only 规则不一致的 backend-import、版权年份范围检查。
部分覆盖提示只搜索 `tests/torch` 路径，不能识别相应 CPU 回归位置。
这些检查没有通过，未删除规则或放宽阈值；本 PR 保持 Draft 等待收敛。

QAT 的通用 selector、named scheme 到策略化 Family 的桥接，以及主线 W4A8 block128 的算子契约问题，
属于后续工作。本 PR 的 MXFP8 通过记录不能替代五种 QAT 配置的验证。

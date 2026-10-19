# Module Family 主线迁移检查点与 QAT 前置验证

日期：2026-10-19。**本报告记录首次未通过 C0 门禁的检查点，不是迁移完成或 QAT 完成报告。**

后续已定位并修复 HCCL 环境和上游 HSDP 视图问题，补跑训练、恢复和重算；最新证据见[故障归因与修复报告](module_family_a5_root_cause_20261019.md)。下文保留首次检查点的历史结果。

## 1. 来源与分支

| 项目 | 值 |
| --- | --- |
| 官方仓库/默认分支 | `mindspore-ai/hyper-parallel` / `master` |
| 启动固定基线 | `ce7d4b0051a6003de76c3401e4de4a6433a09497` |
| fetch UTC 时间 | `2026-10-18T19:41:01Z`（北京时间 2026-10-19） |
| 来源累计范围 | `36a9653b754966efab9cde8dee4068f09b861d11..32e2dd9681ec335126ded48172647e69bb933991` |
| 迁移分支 | `integration/module-family-mainline-20261018T194051Z-1071304` |
| 公共保存修复 | `da9aca5fcd2f6ac5b4da5b871126ae7a2c6e47fb` |
| 迁移候选代码 | `915519dc8658727b1f1ea0c9912894332add6357` |
| 交付前上游查询 | 仍为 `ce7d4b0051a6003de76c3401e4de4a6433a09497` |
| `C0_HEAD` / `QAT_HEAD` | 均未产生：C0 required 验收未通过，QAT 未开始 |

使用独立 worktree `/home/c00913822/hyper-parallel-family-mainline-20261018T194051Z-1071304`。
保留原评审分支、第二阶段分支及原工作树。本地仓库起初为浅仓库，先补齐来源历史，再执行工作包脚本获取官方快照；未以缓存冒充最新。

来源四提交的处理如下，逐文件清单见[31 文件映射](module_family_mainline_source_map.md)。

| 来源提交 | 迁移位置 |
| --- | --- |
| `d7e06241` | 迁移候选：Family、replacement、非融合专家与 EP 接线 |
| `ce530940` | 迁移候选：融合 Function、ops 目录 adapter；`da9aca5f`：物理保存 helper |
| `a3051b75` | 迁移候选：即时权重复制、meta 转换、Trainer 观察器和配置生成器 |
| `32e2dd96` | 迁移候选：五字段规则、配置覆盖与序列化；另修复作用域与注册快照 |

当前差异为 `UPSTREAM_BASE_SHA..迁移候选代码`，**不能把它命名为已验收的 C0 diff**。
没有新增 `use_low_precision` 或 `LowPrecisionGroupedSwiGLU`，没有 QAT 分支或 QAT diff。

## 2. 实际代码变更

- 公共 GroupedLinear 在释放前向专用方向后保存 payload/scale，原始 counts 和有效 offsets 均由 autograd 管理。反向重建临时包装，不清空保留图的数据，使用一次可微边界。
- 保留上游 W8/native W4/fake W4 数值 hooks。MXFP8 的 row-only 量化不再错误传入仅适用于 column 方向的分组参数，修复冻结权重路径。
- 将第一阶段融合 adapter 迁到 `quantization/ops/npu_mxfp8.py`；非融合专家直接组合公共 `MXFP8GroupedLinear`。没有复制旧 GMM 引擎。
- 合并最新配置解析和命名 scheme 注入；Family 只选择类。普通 target 的合法同名构造参数不受其它 Family 影响；注册表建立只读映射快照。
- 保留主线 EP 的 capacity/overlap/custom gate 参数，将 grouped 自动选择改为三态，保留显式 False 的含义。参数分片前在现行 mesh 入口校验旧 MXFP8 范围。
- 保留已有源权重、meta 和转换关系。旧低精专家壳仍是主线 `quantization.modules.GroupedExperts`，未修改其 Parameter 身份或布局。
- 测试适配包括 `functional` → `ops` 导入、主线低精专家类改名、`_ep_use_grouped_gemm` 属性、Qwen 示例模板子目录；没有删除数值断言或放宽阈值。

[完整第一阶段设计](module_family_implementation.md) 已更新为当前调用链。历史带日期报告保留原貌，不作为本轮通过证据。

## 3. 环境与证据

全部证据目录：`/home/c00913822/hp-validation/family-qat-mainline-20261019`，以下记为 `V`。
运行环境详情和导入路径见 `$V/environment.json`；代码指纹见 `$V/candidate-source-hashes.json`。

- 独立 venv：`/home/c00913822/hp-mainline-qat-venv`，editable 指向本次 worktree；复用已有依赖，未升级共享软件。
- Python 3.10.21；PyTorch 2.9.0+cpu；torch_npu 2.9.0.post7+gitddd16dd；Transformers 5.12.0。
- A5 Ascend950PR，驱动工具版本 `25.1.rc1.b087`。
- 主要 CANN：`/home/zhaoyu65/cann-9.2.0/ascend-toolkit/set_env.sh`。
- 环境对照：`/home/c30087003/env/cann/cann-9.2.0/set_env.sh`，只切换子进程环境。
- A5 专家测试使用物理卡0/1，Trainer 使用2，独立 HCCL 对照使用2、5及0/1。避开运行已有服务的4号卡；没有重置设备或结束无关任务。
- 网络操作统一 `http://127.0.0.1:7893`。训练资产由原生成器离线生成。

## 4. 验证结果

| 项目 | 真实结果 | 日志 |
| --- | --- | --- |
| 纯上游相关 CPU 控制 | 58 passed，83 subtests passed | `upstream-cpu.log` |
| 新生命周期修复前 | 18 个失败子项：重复反向、版本检查、物理 hooks；失败证据保留 | `lifecycle-before.log` |
| 公共保存修复后 | 61 passed，101 subtests passed | `lifecycle-final.log` |
| 最终迁移 CPU 集合 | **286 passed，714 subtests passed，0 failed、0 skipped** | `migration-cpu-candidate.log` |
| 原 A5 完整入口 | **5 passed、3 failed、0 skipped**；三项 EP2 都在 HCCL 初始化失败 | `c0-a5-final.log`、`expert-workers/` |
| 首路 BF16 Trainer | 初始模型权重逐值检查通过；第一个 microbatch 后在 loss 汇总 all_reduce 阻塞，未完成 optimizer step | `trainer/ep1-bf16.log` |
| 六路 Trainer、六路新进程恢复、融合 EP1/2 full 重算 | **未完成**；不引用历史结果替代 | 无本轮通过记录 |
| 设计 YAML / Mermaid | 真实 resolver 解析1个片段；Mermaid解析3幅图通过，未作浏览器渲染 | `doc-yaml-check.log`、`mermaid-check.log` |
| compileall / diff检查 | 仅编译29个变更 Python 文件，通过；diff检查通过 | `compile-doc-check.log` |
| 公共保存修复 AutoGit | 通过，含非阻断提示 | `dependency-all-check.log` |
| 迁移 AutoGit | **未通过**，包括主线导入/版权识别、公开文档与测试标记等检查项；仍需处理 | `migration-static-candidate.log` |

单卡5个通过入口覆盖非融合专家、融合与非融合对照、三路loss、融合边界、物理保存/offload/重算。原有数值阈值保持。
完整 A5 初轮曾因冻结分支分组参数、mesh 名称及旧测试属性名称失败，这些已修复；最终三个失败均为通信器初始化错误。各轮日志保留。

另直接调用现有主线策略作单卡前置诊断：

| 策略 | 结果 |
| --- | --- |
| native W8A8 block32 | 两次一阶反向完成，dX/dW 最大绝对差均为0 |
| native W4A8 block32 | 同上 |
| fake W4A8 block32 | 同上 |
| native W4A8 block128 | 前向 GMM 拒绝 FP32 per-token scale，要求 E8M0；两套已安装 CANN 均复现 |
| fake W4A8 block128 | 反向 GMM 报 scale/weight 转置形状不匹配 |

上述是公共策略诊断，**不是 QAT Family 验收**。日志：`lifecycle-a5.log`、`lifecycle-a5-fake.log`、`lifecycle-a5-alternate.log`；脚本 `$V/check_grouped_lifecycle.py`。
CPU 采用真实 autograd/量化 Tensor 包装，替身仅隔离设备算术；不能据此声称 block128 硬件通过。

## 5. HCCL 阻塞证据与未完成项

完全不导入 HyperParallel 的 `$V/check_hccl.py` 仅初始化进程组并执行 all_reduce。
单 rank、双 rank 均失败；另一套已安装 CANN 也失败。错误为 `hcclCommInitRootInfoConfig` code 4，
对应 plog 中有 `load allocated resource to json fail`、`RootInfoDetect failed`。
详见 `hccl-control-valid.log`、`hccl-control-alternate.log`、`hccl-control-ep2.log` 和复制的 plog。
第一次诊断使用了不合法的45秒 HCCL连接超时，已按运行时要求改为120秒重新验证；首次错误不作为环境阻塞证据。

这证明当前通信故障可以在 Family 之外复现，但不代替 EP2 代码正确性的验收。
未通过 required 门禁，故 `C0_HEAD` 保持未定义，按工作包顺序没有开始 QAT Family 实现。
下一步需要可用的 HCCL 环境入口，先完成 C0 全部训练/恢复/重算及静态检查；之后才建立 QAT 分支，处理 block128 契约并实施完整五方案桥接验证。

## 6. 可复现命令

```bash
cd /home/c00913822/hyper-parallel-family-mainline-20261018T194051Z-1071304
source /home/zhaoyu65/cann-9.2.0/ascend-toolkit/set_env.sh
export OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/c00913822/hp-mainline-qat-venv/bin/python
V=/home/c00913822/hp-validation/family-qat-mainline-20261019

$PY -m pytest -q \
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
  tests/ut/trainer/test_optimizer.py

ASCEND_RT_VISIBLE_DEVICES=0,1 HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120 \
  $PY -m pytest -q -s tests/torch/expert_parallel/test_grouped_experts_mxfp8.py

# 使用新的输出目录，避免混入既有 trace。
$PY -m examples.training_demo.prepare_family_validation "$V/trainer-rerun"
ASCEND_RT_VISIBLE_DEVICES=2,3 FAMILY_TRACE_DIR="$V/trainer-rerun/trace-ep1-bf16" \
  $PY -m torch.distributed.run --standalone --nproc-per-node=1 \
  -m tests.torch.expert_parallel.family_trainer_validation "$V/trainer-rerun/ep1-bf16.yaml"

ASCEND_RT_VISIBLE_DEVICES=0,1 HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120 \
  timeout 180 $PY -m torch.distributed.run --standalone --nproc-per-node=2 "$V/check_hccl.py"
```

训练资产为随机小型原生 Qwen3-MoE（2层、H128、I256、4专家、top-k2）及原固定合成文本、WordLevel tokenizer。
没有下载或使用预训练模型/标准数据集；本轮未完成训练，不能提供新的loss轨迹、收敛或性能结论。

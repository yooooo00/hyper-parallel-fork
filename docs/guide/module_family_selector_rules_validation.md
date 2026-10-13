# Module Family 第一阶段 selector 规则验证

日期：2026-10-13。分支：`refactor/module-family-selector-rules`。
唯一基线：`a3051b75250a87ae09a7c7c300b4fa63711ccce2`。
独立 worktree：`/home/c00913822/hyper-parallel-family-selector-rules`。
未合入第二阶段实现；原 `feat/grouped-experts-family-mxfp8` 工作树及历史保持不变。

## 变更范围

| 文件 | 变更 |
| --- | --- |
| `hyper_parallel/components/modules/family.py` | 新增 `SelectorSpec`、`VariantRule`，扩展 Family 声明与注册检查；按完整具名组合选型 |
| `hyper_parallel/trainer/config/replacement.py` | 动态保存显式 selectors、查询 schema 类型，复制时重新选型 |
| `hyper_parallel/trainer/config/resolver.py` | Target 级覆盖按 schema 类型调用现有 `normalize_value()` |
| `tests/ut/components/modules/test_module_family_selectors.py` | 10 项定向测试，含仅测试注册的 str/bool Family |
| `docs/guide/module_family_implementation.md` | 新增第一阶段设计正文，包含 YAML、声明式选型和三幅 Mermaid 图 |
| 本文件 | 独立记录运行环境、命令及结果 |

`resolve_module_family_target(target, arguments) -> (callable, kwargs)` 接口保持不变。
两个已有开关只出现在 GroupedExperts 的 schema/规则声明中，通用代码没有专属名称分支。
YAML API、三个实现类、严格未知关键字检查及配置错误类型保持兼容；非法组合的错误信息
改为列出 Family、实际参数和合法组合。普通构造参数的默认值补齐语义保持不变。

计算、路由、通信、参数布局、checkpoint 转换和融合保存实现，以及所有既有测试文件均未修改。

## 环境

本机 A5 Ascend 950PR，驱动工具显示 `25.1.rc1.b087`。
使用新建的 `/home/c00913822/hp-selector-venv`，复用已有依赖，不修改共享依赖或原 venv。
editable 安装指向本 worktree，并验证 `hyper_parallel.__file__`。

| 项目 | 版本或路径 |
| --- | --- |
| Python | 3.10.21，底座 `/home/zhaoyu65/miniconda3/envs/zhaoyu_dvm` |
| PyTorch | 2.9.0+cpu，实际计算使用 torch_npu |
| torch_npu | 2.9.0.post7+gitddd16dd |
| Transformers / NumPy | 5.12.0 / 1.26.4 |
| CANN 环境 | `/home/zhaoyu65/cann-9.2.0/ascend-toolkit/set_env.sh` |
| 设备 | 专家测试使用物理卡0/1；Trainer使用物理卡2/3 |
| 网络 | 所有网络操作使用 `http://127.0.0.1:7893`；训练资产离线生成 |

运行前 `npu-smi info` 显示无 NPU 运行进程；没有终止无关进程或重置设备。
全部原始输出位于新目录 `/home/c00913822/hp-validation/family-selector-rules-20261013`，
下文记为 `V`。分布式 worker 输出另保存到 `$V/expert-workers`。

## CPU 与静态检查

| 项目 | 结果 | 日志 |
| --- | --- | --- |
| 原6个CPU文件 + 新selector测试 | **62 passed、190 subtests passed；0 failed、0 skipped** | `cpu-final.log` |
| AutoGit check | **通过**，含 pylint、复杂度、Markdown和目录检查 | `autogit-check.log` |
| compileall | 仅检查3个生产文件和1个新增测试文件，通过 | 命令见下文 |
| 文档YAML | 1个配置块通过真实 `resolve_config()`，选中融合类及EP2 | `doc-yaml-check.log` |
| Mermaid | Mermaid 11.12.0解析器验证3幅图通过；未执行浏览器像素渲染 | `mermaid-parse.log`、`diagram-*.mmd` |
| 文档链接、git diff --check | 通过 | 提交前检查 |

AutoGit 的非阻断提示包括：覆盖率路径推断只寻找 `tests/torch/...`，未识别本次真实 UT 路径；
以及与本次 Python/Markdown 变更无关的 C++/Shell 工具安装建议。没有关闭检查或提高阈值。
CPU 的警告主要来自仓库未注册的 pytest marker，未造成 skip。

首次 CPU 的7个子项失败源于新增测试误把普通构造默认参数也当作不得序列化的字段。
原解析器会补入 `context=None`、`module_fqn=""`；修正测试为只约束 selector 显式性，
没有改变这个既有行为。初次日志 `cpu.log` 和后续日志均保留。
新测试沿用原 Family 用例的 CPU 标记及三段式说明；严格类型检查仍区分 bool 与 int。

```bash
cd /home/c00913822/hyper-parallel-family-selector-rules
source /home/zhaoyu65/cann-9.2.0/ascend-toolkit/set_env.sh
export OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/c00913822/hp-selector-venv/bin/python
V=/home/c00913822/hp-validation/family-selector-rules-20261013

$PY -m pytest -q \
  tests/ut/components/modules/test_module_family.py \
  tests/ut/components/modules/test_module_family_selectors.py \
  tests/ut/components/quantization/test_mxfp8_memory.py \
  tests/ut/components/quantization/test_memory_contracts.py \
  tests/ut/components/quantization/test_saved_quantized.py \
  tests/ut/dual_mode_dtensor/test_module_replacement.py \
  tests/ut/dual_mode_dtensor/test_ep_moe_local_region.py

$PY -m compileall -q \
  hyper_parallel/components/modules/family.py \
  hyper_parallel/trainer/config/replacement.py \
  hyper_parallel/trainer/config/resolver.py \
  tests/ut/components/modules/test_module_family_selectors.py

PATH=/home/c00913822/hp-selector-venv/bin:/home/c00913822/hp-venv/bin:/home/c00913822/hp-tools/npm/bin:/home/c00913822/hp-tools/node-v22.14.0-linux-arm64/bin:$PATH \
  $PY .agent/skills/autogit/scripts/autogit.py check
git diff --check
```

## A5 专家测试

原有入口不加外层 torchrun，完整执行结果为 **8 passed、0 failed、0 skipped**，156.04秒。
日志：`$V/a5-experts.log`。本轮没有 HCCL、stream 或设备资源故障，也没有设备失败重试。

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 $PY -m pytest -q -s \
  tests/torch/expert_parallel/test_grouped_experts_mxfp8.py
```

覆盖真实 FP8 GMM、融合前反向、EP2、跨rank路由、空接收rank、零token专家、空输入、
冻结权重、重复反向、三步更新，以及物理 saved-tensor 的 CPU offload/重算。
数值阈值直接沿用基线用例，未修改设备测试或其阈值。

## 六路 Trainer

直接使用基线的 `prepare_family_validation.py` 和 `family_trainer_validation.py`。
模型为真实 Transformers `Qwen3MoeForCausalLM` 的随机小配置：seed42、2层、H128、I256、
4专家、top-k2、vocab128，BF16参数。tokenizer和128行文本是原验证生成器的离线合成资产，
本轮未改用预训练权重或标准语料；用于选择器重构的既有功能回归。

**6个配置均完成10次前向、反向、AdamW更新；0失败、0跳过**。
观察器核对加载后的原生权重与本地参数、专家梯度及真实算子调用。
以下为第10步时每rank的累计调用数：

| EP | 模式 | FP8 GMM | 融合前向 | 融合反向 |
| ---: | --- | ---: | ---: | ---: |
| 1 | BF16 | 0 | 0 | 0 |
| 1 | MXFP8非融合 | 240 | 0 | 0 |
| 1 | MXFP8融合 | 240 | 40 | 40 |
| 2 | BF16 | 0 | 0 | 0 |
| 2 | MXFP8非融合 | 120 | 0 | 0 |
| 2 | MXFP8融合 | 120 | 20 | 20 |

同拓扑、同批次的10步 CE 最大绝对差异如下。此表仅记录观察结果，不设事后收敛阈值：

| EP | 融合−非融合 | 融合−BF16 | 非融合−BF16 |
| ---: | ---: | ---: | ---: |
| 1 | 0.000271559 | 0.000530958 | 0.000468493 |
| 2 | 0.000608206 | 0.000552893 | 0.000630379 |

原比较器执行成功，完整轨迹在 `$V/trainer-comparison.json`；各命令、卡号和退出码
在 `$V/trainer/commands.jsonl`，日志和每rank快照位于 `$V/trainer`。
本轮未额外复跑 Trainer 新进程恢复或 full 重算；不据已有历史结果宣称本轮通过。
原 A5 算子入口中的保存/offload/重算已实际重跑。

以下命令需使用新的输出目录，避免混合旧trace：

```bash
T=$V/trainer
$PY -m examples.training_demo.prepare_family_validation "$T"
for ep in 1 2; do
  for mode in bf16 mxfp8 fused; do
    ASCEND_RT_VISIBLE_DEVICES=2,3 FAMILY_TRACE_DIR="$T/trace-ep$ep-$mode" \
      $PY -m torch.distributed.run --standalone --nproc-per-node="$ep" \
      -m tests.torch.expert_parallel.family_trainer_validation "$T/ep$ep-$mode.yaml"
  done
done
$PY -m tests.torch.expert_parallel.compare_family_training "$T"
```

## 文档与范围核对

[第一阶段设计](module_family_implementation.md) 已将声明式规则融入配置、数据结构与构建图，
并保留继承图、EP路径、权重及物理保存机制。设计正文不包含本机安装记录、调试历史或性能推断。
本轮未新增 `FamilySelection`、默认并行adapter、多Family设备实现、TP/CP、EP>2、
新融合kernel或根配置列表CLI语法，也未改动原有计算、权重及生命周期文件。

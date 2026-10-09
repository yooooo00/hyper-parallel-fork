# GroupedExperts Family：A5 真实训练验收记录（2026-10-09）

本记录补充 2026-10-09 设计文档的第 8.3 节。测试基线为
`ce53094070f8566d872e1c0377e98e246e4e6a39`，上游比较基线为
`36a9653b754966efab9cde8dee4068f09b861d11`。新增测试的结果单独记录，
不把 9 月的运行结果合并为本次全量通过。修复后的结果对应本报告同一提交中的代码。

## 环境来源

训练命令在本机执行，没有通过 Docker 启动。Python 环境由三部分组成：

| 层次 | 当前来源 |
| --- | --- |
| 基础 Python / 算子依赖 | `/home/zhaoyu65/miniconda3/envs/zhaoyu_dvm` |
| 本任务 venv | `/home/c00913822/hp-venv`，`include-system-site-packages = true` |
| CANN | `/home/zhaoyu65/cann-9.2.0/ascend-toolkit/set_env.sh` |
| HyperParallel | `/home/c00913822/hyper-parallel`，editable 源码安装 |

基础 Conda 环境的 `conda-meta/history` 记录于 2026-09-23，通过
`conda env create -f zhaoyu_dvm.yml` 创建。未找到该原始 YAML，不能声称
已恢复完整制作配方。Conda 历史也不能证明之后 pip 安装的完整来源。
外层 venv 继承基础环境依赖，所以它不是独立、不可变的环境镜像。

本次设备是 Ascend950PR_9589，共 8 张可见卡；测试主要使用 0、1 卡。
Python 3.10.21，PyTorch 2.9.0+cpu，torch-npu 2.9.0.post7+gitddd16dd，
Transformers 5.12.0，NumPy 1.26.4，pytest 9.1.1，CANN 9.2.0-beta.2。
PyTorch 的 `+cpu` 包版本后缀不意味着这些用例在 CPU 上计算：NPU 运算由
torch-npu 注册，测试实际检查 FP8 GMM 和融合前后向的调用。

完整 Trainer 首次启动发现缺少 torchdata、datasets；这与此前仅验收专家
算子不同。补充依赖安装在本任务 venv 中，网络使用 `127.0.0.1:7893` 代理。
本机没有已有预训练模型、tokenizer 或语料，因此验证资产在本地生成。

本次新增依赖为 torchdata 0.11.0、datasets 4.4.2、pybind11 3.0.1，
连带安装 pyarrow 25.0.1 等；完整清单位于证据目录的 `pip-freeze.txt`。
`pip check` 仍报告若干继承环境的 CANN 工具可选依赖缺失，以及 OpenCV
对 NumPy 版本的声明冲突。本次没有调用这些工具，也没有宣称环境整体
依赖检查通过；已执行的训练、量化及 checkpoint 路径以实际结果为准。

本次补充训练依赖的命令如下。它们不是最初 `zhaoyu_dvm.yml` 的替代品。

```bash
export HTTP_PROXY=http://127.0.0.1:7893 HTTPS_PROXY=http://127.0.0.1:7893
export ALL_PROXY=http://127.0.0.1:7893
export http_proxy=http://127.0.0.1:7893 https_proxy=http://127.0.0.1:7893
export all_proxy=http://127.0.0.1:7893 NO_PROXY= no_proxy=
/home/c00913822/hp-venv/bin/python -m pip install torchdata==0.11.0 datasets==4.4.2
/home/c00913822/hp-venv/bin/python -m pip install pybind11==3.0.1
```

## 已完成的固定基线重跑

- CPU：50 项测试、137 项子测试通过。
- 原有真实设备 6 个入口首次运行：5 通过、非融合 EP=2 失败。
  失败日志包含 HCCL/AICPU 执行异常及 SIGABRT；相同代码单独重跑该入口通过。
  此次异常尚未完成根因归属，不能据单次重跑通过宣称已经消除。
- 原始证据目录：`/home/c00913822/hp-validation/trainer-20261009`。

## 融合边界与保存生命周期

新增固定 projected/upstream 的前后向对照，counts 为 `[1, 0, 31, 32, 33]`。
逐项记录 row/column payload、scale 原始字节差异，以及反量化后的数值误差。
列向 scale 的物理存储带有专家间及尾部 padding，不能用整个 buffer 的
字节不等数量解释有效量化数值的误差。测试使用可精确表示的分组常量，
独立核对列向解码的位置后再比较随机输入。

固定输入的行向反量化对照：融合相对非融合的前向 NRMSE 为 0.0144422，
反向为 0.0104854。融合相对未量化 BF16 边界值分别为 0.0268904、0.0264520。
这些证据说明部分差异出现在激活/量化边界，并不证明全部误差都源于某个
具体舍入步骤。阈值沿用原测试：RMSE ≤ 0.1 × 参考 RMS + 1e-6，
最大绝对误差 ≤ 0.5 × 参考最大绝对值 + 1e-6；没有依据本次结果放宽。

新增真实 A5 融合测试覆盖 `save_on_cpu(pin_memory=False)`、
非重入 `checkpoint` 及保留图后的重复反向。输出、输入梯度及两组权重梯度
与直接融合路径逐元素相等。该结果不覆盖 pinned-memory 异步 offload、
Trainer attention swap 或任意分布式重算组合。

## 完整 Trainer 验证资产

`examples/training_demo/prepare_family_validation.py` 离线生成真实
`Qwen3MoeForCausalLM` 类的随机初始化原生 HF 权重、本地 WordLevel tokenizer、
固定合成文本及六份完整 YAML。规模为两层、E=4、H=128、I=256、top-k=2，
覆盖 EP=1/2 下的 BF16、非融合 MXFP8、融合 MXFP8。

它通过 `HyperAutoModelForCausalLM.from_pretrained` 加载本地随机权重，沿用
生产配置解析、模型 recipe、planner、TextTrainer 和 optimizer/checkpoint。
这里的 `from_pretrained` 是加载接口名称，资产不是经过预训练的模型。
这类短程 CE 训练可以验证训练链路，不能证明真实预训练模型的收敛质量。

`tests/torch/expert_parallel/family_trainer_validation.py` 仅增加观测与断言：
记录实际类、本地专家权重形状、真实算子调用、专家梯度、样本哈希和每步权重。
它不替换 Router、通信、loss 或 optimizer 的数值计算。

## 本次发现并修复的问题

单卡 HF `from_pretrained` 会先加载真实权重，再执行 replacement。
原 `GroupedExperts._initialize_weights()` 在需要转置时只分配空参数，
没有复制已加载的源值；单卡非 meta 路径又不会重新加载 checkpoint。
这使真实 Trainer 的 BF16 与融合路径都出现了非有限专家梯度。

修复在高性能基类中完成：对非 meta 源参数复制可用权重的转置；meta
构建继续由原来的 `make_transforms()` 和 checkpoint loader 填充。
重命名的 baseline bias 同样保留值和冻结状态。低精 Family 仍不支持专家 bias。
没有增加第二套持久权重，也没有修改旧低精类的公开布局。

此前测试辅助函数会手动执行 checkpoint 转换并复制权重，掩盖了真实单卡
构建的缺口。本次补充直接走生产 replacement executor、无辅助重载的 CPU
回归用例；真实 Trainer 在训练前还逐个核对原生 HF 权重和最终本地参数，
包含拆分专家权重的拼接、转置及 EP 分片。

## 修复后真实训练结果

六组均运行相同原生初始权重、固定文本、真实 tokenizer/data pipeline、
CE、AdamW 和 cosine scheduler，各完成 10 次优化器更新。TP=CP=PP=1。
构建、路由、分片、loss 和 optimizer 均使用生产实现，未以 mock 替代。

| 模式 | 第 1 步 CE | 第 10 步 CE |
| --- | ---: | ---: |
| EP=1 BF16 | 4.868292570 | 4.484850407 |
| EP=1 非融合 MXFP8 | 4.868127108 | 4.484700441 |
| EP=1 融合 MXFP8 | 4.868099928 | 4.484708786 |
| EP=2 BF16 | 4.867922783 | 4.484517574 |
| EP=2 非融合 MXFP8 | 4.867793083 | 4.484244823 |
| EP=2 融合 MXFP8 | 4.868056536 | 4.484853029 |

EP=2 表中使用两个 rank 的记录均值。每种模式都验证了有限、非零专家梯度，
EP=2 每个 rank 的局部专家数为 2；两条低精路径都有实际 FP8 GMM 调用，
融合路径同时有实际融合前向和反向调用。三种精度逐 rank、逐步的输入及标签
哈希相同。没有要求 EP=1 与 EP=2 的量化分组结果逐位相等。

| 10 步内最大 CE 相对差 | EP=1 | EP=2 |
| --- | ---: | ---: |
| 融合 / 非融合 MXFP8 | 0.006053% | 0.013563% |
| 融合 / BF16 | 0.011116% | 0.012272% |
| 非融合 MXFP8 / BF16 | 0.009808% | 0.013198% |

这些 CE 差异是观测值，不是事后设定的收敛验收阈值。
原有 10% RMS / 50% 最大误差阈值用于专家输出、梯度和激活量化边界，
不应把它套用为语言模型训练质量标准。

六组都在第 5、10 步保存分布式 checkpoint，随后分别启动新进程从第 5 步
恢复到第 10 步。逐 rank 检查得到：恢复后的 CPU/NPU/Python RNG 与第 5 步一致；
第 6–10 步的 CE、全局梯度范数、所有参数、输入/标签哈希及 RNG 与连续训练
完全相等，参数比较使用 `rtol=atol=0`。这覆盖实际 optimizer/scheduler 和
数据进度恢复的联合行为，并非仅能读出文件。

此外，融合模式在 EP=1/2 各运行 10 步 `activation_checkpoint.mode=full`。
实际融合前向调用增加，确认发生重算；每一步 CE、梯度范数、所有参数、样本
哈希及 RNG 与关闭重算的对应运行完全相等。

修复后相关 CPU 集合：**52 passed，140 subtests passed**；
真实设备完整 launcher 集合：**8 passed，155.94 秒，无 skip**。
正式训练、续训与 full 重算合计执行 110 次优化器更新，不含诊断和失败重试中的更新。

## 设备运行时异常与适用边界

本轮没有修改驱动或重置设备，也没有中止其他用户的进程。
物理卡 0/1 上记录了三类未完成根因归属的事件：初始 EP 测试 HCCL/AICPU
异常、单卡融合启动时 `halSqCqAllocate` / stream 分配错误、一次双卡非融合
任务在模型加载后的初始化阶段停留约 2 分半。前两项相同代码重跑通过，
最后一项保留日志后终止本任务进程，在物理卡 2/3 上完成对应训练。

六组原始训练中，EP=1 三组及 EP=2 BF16 使用 0/1；EP=2 两组 MXFP8 使用 2/3。
六组续训使用 2/3，最终 8 个设备测试使用 4/5，full 重算使用 6/7。
不能用后续成功覆盖这些异常，更不能宣称本机驱动稳定性已得到保证。

尚未验证的边界包括：真实预训练模型与真实语料的收敛/评估、FP16、
独立专家 DP、FSDP 分片度大于 1、HSDP 拓扑扩展、attention swap、
pinned-memory 异步 offload，以及代表性大 MoE 的独占性能和内存剖析。
当前 EP=2 Trainer 确实经过 HSDP 包装和 dense 参数同步，但 shard degree 为 1，
不能据此声称完成所有 FSDP/HSDP 组合验收。TP/CP/PP 和 EP>2 的保护没有移除。
Trainer 的采样/权重观测会同步设备，日志中的耗时不是性能 benchmark。

## 复现命令与证据

以下为本轮命令形式；重跑时选择新的 `family_root`，不要向旧 trace 追加结果。
完整原始证据根目录为 `/home/c00913822/hp-validation/trainer-20261009`，
最终六路对照目录为其下的 `matrix`。

```bash
cd /home/c00913822/hyper-parallel
source /home/zhaoyu65/cann-9.2.0/ascend-toolkit/set_env.sh
export OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export ASCEND_RT_VISIBLE_DEVICES=2,3
family_python=/home/c00913822/hp-venv/bin/python
family_root=/home/c00913822/hp-validation/trainer-20261009/matrix

$family_python -m examples.training_demo.prepare_family_validation "$family_root"
for family_ep in 1 2; do
  for family_mode in bf16 mxfp8 fused; do
    export FAMILY_TRACE_DIR="$family_root/trace-ep$family_ep-$family_mode"
    $family_python -m torch.distributed.run --standalone --nproc-per-node="$family_ep" \
      -m tests.torch.expert_parallel.family_trainer_validation \
      "$family_root/ep$family_ep-$family_mode.yaml"
    export FAMILY_TRACE_DIR="$family_root/resume-ep$family_ep-$family_mode"
    $family_python -m torch.distributed.run --standalone --nproc-per-node="$family_ep" \
      -m tests.torch.expert_parallel.family_trainer_validation \
      "$family_root/ep$family_ep-$family_mode.yaml" \
      --checkpoint.restore_from="$family_root/ep$family_ep-$family_mode-ckpt/global_step_5" \
      --checkpoint.checkpoint_dir="$family_root/resume-ep$family_ep-$family_mode-ckpt"
  done
  export FAMILY_TRACE_DIR="$family_root/ac-ep$family_ep-fused"
  $family_python -m torch.distributed.run --standalone --nproc-per-node="$family_ep" \
    -m tests.torch.expert_parallel.family_trainer_validation \
    "$family_root/ep$family_ep-fused.yaml" --activation_checkpoint.mode=full \
    --checkpoint.checkpoint_dir="$family_root/ac-ep$family_ep-fused-ckpt"
done
$family_python -m tests.torch.expert_parallel.compare_family_training \
  "$family_root" --resume --activation-checkpoint
$family_python -m pytest -q -s tests/torch/expert_parallel/test_grouped_experts_mxfp8.py
```

CPU 集合命令：

```bash
$family_python -m pytest -q \
  tests/ut/components/modules/test_module_family.py \
  tests/ut/components/quantization/test_mxfp8_memory.py \
  tests/ut/components/quantization/test_memory_contracts.py \
  tests/ut/components/quantization/test_saved_quantized.py \
  tests/ut/dual_mode_dtensor/test_module_replacement.py \
  tests/ut/dual_mode_dtensor/test_ep_moe_local_region.py
```

主要证据：`environment.json`、`pip-freeze.txt`、`pip-check.txt`、
`cpu-final.log`、`devices-final.log`、`matrix/full-comparison.json`、
各 `trace-*` / `resume-*` / `ac-*` 中的逐 rank JSONL 和逐步参数快照。
依赖安装日志及所有诊断失败日志保留在同一证据根目录。

## 改动文件

| 文件 | 用途 |
| --- | --- |
| `hyper_parallel/components/modules/grouped_experts.py` | 保留已加载投影及重命名 bias 的值；补充接口文档 |
| `tests/ut/components/modules/test_module_family.py` | 不借助辅助重载的真实 replacement 回归 |
| `tests/torch/expert_parallel/_test_grouped_experts_mxfp8.py` | 融合前后向边界、物理保存、CPU offload、重复反向及重算测试 |
| `tests/torch/expert_parallel/test_grouped_experts_mxfp8.py` | 新增真实设备测试入口 |
| `examples/training_demo/prepare_family_validation.py` | 生成离线原生模型、tokenizer、文本和完整配置 |
| `tests/torch/expert_parallel/family_trainer_validation.py` | 观测生产 Trainer，校验原生权重、分片、梯度和真实算子调用 |
| `tests/torch/expert_parallel/compare_family_training.py` | 六路 CE、精确恢复和 full 重算比较 |
| `docs/guide/grouped_experts_mxfp8_a5_validation.md` | 本次中文实测报告 |
| `docs/guide/grouped_experts_mxfp8_fusion.md` | 链接本次修复与实测，保留历史结果适用范围 |
| `.codespellrc` | 将官方软件栈名称 CANN 加入拼写检查词表 |

Python lint、复杂度、Markdown、拼写、代码规范和 AGENTS catalog 检查通过。
自动覆盖率检查仅按预设目录名查找测试，因此仍给出提示；实际测试位于上表列出的
Family UT 和 expert_parallel ST 目录。C++/CMake/shell 检查工具不在本次 Python
与文档变更的检查范围中。Node.js 22.14.0 和 markdownlint-cli2 0.20.0 安装在
`/home/c00913822/hp-tools`，与训练 Python 环境分开。

# RFC #793：A5 实现与验证记录

日期：2026-10-10。代码从 `feat/grouped-experts-family-mxfp8` 的
`a3051b75250a87ae09a7c7c300b4fa63711ccce2` 继续开发；本轮没有刷新或重置上游。
新增代码及可执行验收入口的最终提交为 `c38d9d416821af026759cd5746dd8cce12937cd4`；
各独立提交登记在 [完成清单](RFC793_COMPLETION.md)。
实现契约见 [设计文档](module_family_implementation.md)。

**这份报告不把 RFC 标为全部完成。** DSA 缺真实稀疏算子依赖；GQA 融合示例修订尚无
owner 接受记录；Qwen3-MoE 的一个 MXFP8 对 BF16 梯度比较超出预设阈值。
训练能运行、通信数学正确和低精数值达标分别记账。

## 环境与资产

| 项目 | 实际环境 |
| --- | --- |
| 设备 | 8 张 Ascend 950PR（A5）；共享机器，不占满设备、不重置设备或停止无关进程 |
| Python | `/home/c00913822/hp-venv/bin/python`，Python 3.10.21 |
| 环境来源 | 本机 venv，继承 `/home/zhaoyu65/miniconda3/envs/zhaoyu_dvm` 的包；没有在 Docker 内训练 |
| PyTorch / torch_npu | `2.9.0+cpu` / `2.9.0.post7+gitddd16dd`，真实计算设备为 NPU |
| CANN | `/home/zhaoyu65/cann-9.2.0/ascend-toolkit`，9.2.0-beta.2 |
| Transformers / NumPy / einops | 5.12.0 / 1.26.4 / 0.8.1 |
| 网络 | 网络访问均经 `http://127.0.0.1:7893`；训练使用本地资产及离线开关 |

本轮仅补装 einops 0.8.1，没有修改共享 Conda 底座。没有找到底座最初制作的完整安装脚本，
因此只能记录现有路径、版本和继承关系，不能声称可由某个 Dockerfile 完整复现。
`pip-freeze.txt` 保存包清单。`pip-check.txt` 仍报告既有 OpenCV 5.0.0.93 要求 NumPy>=2
与当前 NumPy 1.26.4 的冲突；该包不参与本次训练，没有为消除无关警告更换 NumPy。

下文 `V=/home/c00913822/hp-validation/rfc793-20261010`，`T=$V/trainer`。
目录中的 `commands.jsonl`、每次运行的 `.log`、每个 rank 的 `*-build.json`、`*-plan.txt`、
逐步 JSONL 和权重快照是原始证据。大型 checkpoint/快照保留在本机，不提交到 Git。

训练模型使用 Transformers 的真实 `Qwen3MoeForCausalLM` 类，**随机初始化，不是预训练模型**：
seed=42，2 层，H=128，I=256，4 个专家，top-k=2，4 个 Q heads、2 个 KV heads，head_dim=32，
vocab=151680，BF16 参数，dropout=0、cache 关闭。缩小配置用于工程验收，不是官方生产规格。
优化器 AdamW：lr=1e-4、weight_decay=.01、betas=(.9,.999)、eps=1e-8，cosine scheduler。

| 资产 | 来源与固定版本 |
| --- | --- |
| tokenizer | 官方 `Qwen/Qwen3-30B-A3B`，revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`；只下载 tokenizer 等小文件，没有下载预训练权重 |
| 数据 | `Salesforce/wikitext` 的 `wikitext-2-raw-v1`，revision `b08601e04326c79dfdd32d625aee71d232d685c3` |
| 本地数据 | `/home/c00913822/hp-validation/datasets/wikitext-2-raw-v1`；train/validation/test 分别 36718/3760/4358 行，包含原始空行 |
| 训练 JSONL SHA256 | `c3fe42e9cc7d482dc79b1812a7c3dd5d3b8c1079abcae08c0846cacf86b749f1` |
| tokenizer.json SHA256 | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |

数据是标准公开语料；训练配置与截取方式是本项目的验收设置。
每行末尾加 EOS，最长窗口 128，连续 position ids，不进行文档内位置重置或 packed mask 重置。
实际首步两个 microbatch 全局长度分别为 128、96，因此 CP2 也覆盖局部长度 48。
清单见 `$T/assets-manifest.json`；36 个生成配置均经过真实配置解析及序列化往返，重新生成的随机权重哈希一致。

## 数值口径

阈值在运行前固定，未根据结果放宽。

| 对照 | 口径 |
| --- | --- |
| 既有 Experts BF16/MXFP8/fused | 沿用原 `test_grouped_experts_mxfp8.py` 及 worker 中的阈值；不改旧用例内容 |
| GQA baseline→MXFP8 的 output/dX/dW | 延用局部 MXFP8 测试的 NRMSE≤.10，最大绝对误差≤参考最大绝对值的 .5，加 1e-6 数值余量 |
| GQA 普通精度并行层级 | `torch.testing.assert_close(rtol=.03, atol=.003)`；与低精阈值分开 |
| 完整 BF16 Trainer 单卡→TP2 | 每项参数分片 relative L2≤.03 且 max_abs≤.003；同批次、更新前梯度 |
| MLA/GDN/KDA CP1→CP2 | FP32，逐元素 `rtol=3e-4, atol=3e-5`，包括输入及参数梯度 |
| 同量化分片的 CP 通信 oracle、默认/显式等价、同拓扑恢复、full 重算 | 对相应输出/梯度或逐步状态要求完全相等，`rtol=atol=0` |
| 训练 CE 差异 | 仅报告绝对/相对差异；不把局部算子的 10% 容差当作收敛或训练质量标准 |

原 native BF16 三个独立 Q/K/V 投影与 packed baseline 的 `rtol=.01, atol=.0001` 严格逐元素
比较存在未达标项，未改写为通过。Qwen dX NRMSE 约 .005446；CPU FP32 的权重转换、输出及梯度
等价测试通过。不同 GEMM/RoPE 路径的误差与 baseline→MXFP8 分开保留在
`/home/c00913822/hp-validation/gqa-numerics.json`。

## 分层结果

统一 CPU 回归：**171 passed、300 subtests passed**，19.81 秒；
`$V/cpu-final-fixed.log`。覆盖旧配置/Experts/物理保存、Family/defaults、CLI、GQA、R7、
TP adjoint/padding/checkpoint 加载和 CP 协议。最终审查补入 SDPA 的 `is_causal=None`
回退测试，真实计算保留，仅 mock 通信。版权 checker 新 fixture 的两次子项失败修复后重新统一运行，
旧失败保留在 `cpu-final.log`，没有用旧数量替代最终结果。

| 层次 | 本轮结果与证据 |
| --- | --- |
| 原 Experts 单卡/EP2 设备入口 | 8 个入口重跑通过，165.77 秒；`$V/experts-eight-regression.log`，含融合、保存/offload、空输入/空 rank、梯度与重算 |
| 原六路 Trainer | EP1/2 × BF16/MXFP8/fused，各 10 步通过；`$V/old-trainer-regression`，比较结果 `$V/old-trainer-comparison.json`；原有本地合成样本保留，仅作旧回归 |
| GQA 单卡 | Llama/Qwen3-MoE 的投影数值、3 次优化器更新、冻结、重复反向、offload/full 重算和空输入通过；`gqa-npu-final.log`、`gqa-npu-empty.log` |
| GQA TP/CP | 两个模型的 TP2、CP2、TP2×CP2，default/explicit adapter、padding/custom mask、TP bias 均有通过记录；`gqa-tp-cp-four.log`、`gqa-masks-validate-bias.log` 等 |
| TP validate | CPU/设备入口验证包含 FQN 的构建错误；RoPE/DTensor 分派仍不支持，不能记作 validate 模式数值通过 |
| EP4 | 三种 Experts 变体均通过实际 planner 的四 rank 分片和原 recipe；每 rank 一个本地专家，覆盖跨 rank 路由及 rank3 无接收 token；`grouped-ep-four*.log` |
| MLA/GDN/KDA | 各 default/explicit CP2 相对 CP1 的前后向通过，共 6 个不同入口；不是一次集合无重试全部通过，详见 `$V/R7-evidence.txt` |
| DSA | 两个真实设备入口因缺 `omni_training_custom_ops` 明确 skip；CPU 隔离可选 import 不构成真实稀疏算子通过 |

上表未带 `$V` 的日志位于 `/home/c00913822/hp-validation`；分布式 worker 日志位于仓库
`logs/tests/torch`。R7 日志另复制到 `$V/r7-workers`。

R7 最终通过用例的最大绝对误差如下；实际断言仍使用上述逐元素相对+绝对容差，
不能把这些观测值重新当作阈值：

| Family | output | dX | dW |
| --- | ---: | ---: | ---: |
| MLA | 1.19209e-7 | 4.17233e-7 | 4.76837e-6 |
| Gated DeltaNet | 0 | 5.96046e-8 | 6.86646e-5 |
| Kimi Delta Attention | 0 | 9.53674e-7 | 8.86917e-5 |

完整数值及 worker 路径见 `$V/r7-metrics.json`。DSA 的 main attention、indexer dW 和辅助 KL
在 CP 下的全局归一化均仍需真实依赖环境验收。

### 完整 TP 梯度对照

新入口 `tests/torch/attention/rfc793_tp_gradient_validation.py` 在生产 clip 前观察真实梯度，
不改 backward、缩放或 optimizer。21 项参数 × 2 个 rank 共 42 个分片全部通过固定阈值：
relative L2 最大 **.0134028**，max_abs 最大 **.000732422**；单卡和两个 TP rank 的梯度范数
均为 **2.265625**。首步 CE：单卡 11.9814873，TP2 11.9815936，输入 batch 哈希相同。
证据：`/home/c00913822/hp-validation/tp-gradient-entry-report.json`。

修复前完整模型的 qkv/q_norm/input_norm 梯度明显不同；原因是独立 TP 模块的复制梯度规则
与 HSDP 源布局归约被叠加，以及纯 TP 未启用 HSDP 同步。现在由 builder 明确传递梯度契约，
摘要标记 `source_layout` 或 `replicated`；通用 collective 语义不变。
此外修复 plain TP 参数 checkpoint 加载的源布局视图、meta 后布局保留和分片词表 padding id。

### 默认 NPU Attention 的边界问题

最终 review 另发现显式 `is_causal=None` 被 CP 解释为 false；修复为沿用模块默认值，
与 HF SDPA 保持一致。CPU 使用真实 SDPA 对照全局后半序列，覆盖 causal/noncausal 两种默认值；
NPU 接口的参数归一化同步修正。缺省参数路径不受此修正影响。

CP 本地 Q48/全局 K96 的原默认 `next_tokens=0` 会额外截断显式全局 causal mask 允许的 key。
独立真实 NPU probe 的 output NRMSE 为 .32335；完整 BF16 和融合 Trainer 均出现过非有限梯度。
修复为矩形 Q/K 默认覆盖完整 K 窗口，显式窗口设置优先级不变。默认/显式无限窗口的
output/dQ/dK/dV 精确对照 2 项通过，CPU 窗口契约 3 项/8 子项通过。
相同实际 microbatch 的 Trainer 修复后输出及梯度有限。

强化到全局 96 token 的 GQA CP2 数值集仍有 **1 项失败**：Qwen3-MoE MXFP8 的
`q_norm.weight` 梯度 NRMSE **.1158468859 > .10**。该用例 output/dX/QKV-dW/o_proj-dW
分别为 .0684141/.0771644/.0772065/.0684310。Llama BF16/MXFP8、Qwen BF16 及两个 MLA
入口通过，合并运行结果 5 passed、1 failed。

以相同 MXFP8 token 分片/量化顺序在单设备拼接 K/V 的无通信 oracle 再对照，两个 rank 的
output/dX/全部 dW 均精确相等。该证据排除了本用例的 CP 分发/重复归约差异，
**不能消除 MXFP8 对 BF16 的误差失败**。原阈值不变，失败保留为验收缺口。
细节见 `gqa-cp-window-analysis-20261010.txt`、`gqa-window-cp-mla-final.log`、
`gqa-window-cp-quantized-reference.log`。

## 真实语料 Trainer 矩阵

三种模式分别为：`bf16` 两个 Family 都使用 baseline；`mxfp8` 两者均低精且 Experts 非融合；
`fused` 为 GQA MXFP8 投影 + Experts 融合 MXFP8。GQA 的融合开关始终 false。

| TP | CP | EP | rank 数 | 每模式步数 | BF16 / MXFP8 / fused |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1 | 1 | 1 | 10 | 全部通过 |
| 2 | 1 | 1 | 2 | 10 | 全部通过 |
| 1 | 2 | 1 | 2 | 10 | 全部通过 |
| 1 | 1 | 2 | 2 | 10 | 全部通过 |
| 2 | 2 | 1 | 4 | 10 | 全部通过 |
| 2 | 1 | 2 | 2 | 10 | 全部通过 |
| 1 | 2 | 2 | 2 | 10 | 全部通过 |
| 2 | 2 | 2 | 4 | 50 | 全部通过 |

共 **24** 个完整矩阵配置通过。另运行 `tp1-cp2-ep1-fused-fsdp2` 和
`tp1-cp2-ep2-fused-explicit` 各10步，合计26个配置。前者实际 dense FSDP mesh
为 `(replicate=1, shard=2, tp=1)`，由构建记录确认真正的 shard=2。
后者两层 GQA 的 `adapter_source=explicit`，默认配置对应 `family default`。

混合222的实际 dense mesh 为 `(dp=1, cp=2, tp=2)`，expert FSDP mesh 为
`(edp_replicate=2, edp_shard=1, ep=2)`；每 rank 本地专家数2。
本地 GQA QKV/o_proj 权重分别为 `[128,128]`、`[128,64]`，Experts 为
`gate_up=[2,128,512]`、`down=[2,256,128]`。
融合50步每 rank 实际调用 FP8 GMM **1200** 次、FP8 Linear **1200** 次、
融合前向/反向各 **200** 次。各拓扑每步各 rank 的全局梯度范数均一致。

混合222的 rank0 CE（首步→第50步）及逐精度最大差异见下表；
跨拓扑的数据并不总相同（EP单独开启时派生 DP 域不同），不据跨拓扑 loss 排名。

| 模式 | 首步 CE | 第50步 CE |
| --- | ---: | ---: |
| bf16 | 11.981532097 | 11.752559662 |
| mxfp8 | 11.981560707 | 11.750691414 |
| fused | 11.981729507 | 11.751529217 |

同拓扑同批次、全部 rank/step 的最大 CE 绝对差异：

| TP/CP/EP | MXFP8−BF16 | fused−BF16 | fused−MXFP8 |
| --- | ---: | ---: | ---: |
| 1/1/1 | 0.002756596 | 0.002615452 | 0.001069069 |
| 2/1/1 | 0.002508163 | 0.002828598 | 0.000547886 |
| 1/2/1 | 0.002860069 | 0.002753735 | 0.001114368 |
| 1/1/2 | 0.001444817 | 0.001242638 | 0.001531601 |
| 2/2/1 | 0.003787041 | 0.003026485 | 0.000922203 |
| 2/1/2 | 0.002943516 | 0.003067970 | 0.000493526 |
| 1/2/2 | 0.002801418 | 0.002087593 | 0.000795364 |
| 2/2/2 | 0.002991676 | 0.003594875 | 0.001429081 |

222的 fused−MXFP8 最大相对 CE 差异为 **.01204%**，fused−BF16 为 **.03059%**。
所有数字都仅描述这次随机小模型训练，未引入事后质量阈值。完整轨迹见 `$T/final-comparison.json`。

恢复 **11 个配置 / 21 个 rank**、full 重算 **5 个配置 / 11 个 rank** 全部精确通过：

- 恢复：111三模式、112三模式、121/122/211/222的融合，以及121融合FSDP2。
- full 重算：111/122/211/222融合，以及121融合FSDP2；10步用例每rank额外40次融合前向，50步用例额外200次。
- 222从第5步恢复至第50步，逐步参数、训练状态、数据和loss均与连续运行完全相等。
- 122融合的显式CP配置与Family默认配置，两rank、10步的全部参数、训练状态和算子计数完全相等。

上述精确比较由仓库比较器执行，原始输出为 `$T/final-comparison.json`，stderr为空、退出码0。

每个构建均精确核对原始 checkpoint 经 packed/转置/分片后的参数值；记录实际 dense mesh、
expert mesh、本地权重形状和 Family 摘要。每步要求所有受观察的可训练投影/Experts 参数
梯度存在且有限，并逐模块统计真实 FP8 Linear/GMM、融合前后向；空 token 模块允许零次算子，
但仍须有合法梯度。检查没有直接 mock replacement、planner、路由或设备计算。

恢复比较读取既有 optimizer slots、parameter groups 和主权重生成摘要，
不调用可能通过 DCP 初始化状态并执行 step 的 optimizer.state_dict。
同拓扑新进程从第 5 步 checkpoint 恢复，对齐第 6 步以后的 loss、grad_norm、数据/RNG、
optimizer/scheduler 和每个本地参数；full 重算还要求真实融合 forward 次数增加。

## 可复现入口

以下命令从仓库根目录执行。先选空闲卡；`0,1,3,4` 是最终四卡矩阵实际使用的物理设备，
不保证将来运行时仍空闲。即便 CPU 比较器导入 torch，也须先加载本机 CANN 环境。

```bash
source /home/zhaoyu65/cann-9.2.0/ascend-toolkit/set_env.sh
export OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/c00913822/hp-venv/bin/python
V=/home/c00913822/hp-validation/rfc793-20261010
T=$V/trainer
```

重新生成时使用一个不存在的新输出目录；生成器不覆盖旧资产：

```bash
$PY examples/training_demo/prepare_rfc793_validation.py /path/to/new-validation \
  --tokenizer-path "$V/tokenizer" \
  --data-path /home/c00913822/hp-validation/datasets/wikitext-2-raw-v1/train.jsonl
```

运行某个完整配置，rank 数为实际 dense/expert mesh 所需数量。
TP=CP=EP=2 使用 4 rank，EP 不在 dense 数量上再乘一次：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,3,4 FAMILY_TRACE_DIR="$T/trace-tp2-cp2-ep2-fused" \
$PY -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m tests.torch.attention.rfc793_trainer_validation "$T/tp2-cp2-ep2-fused.yaml"

ASCEND_RT_VISIBLE_DEVICES=0,1,3,4 FAMILY_TRACE_DIR="$T/resume-tp2-cp2-ep2-fused" \
$PY -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m tests.torch.attention.rfc793_trainer_validation "$T/tp2-cp2-ep2-fused.yaml" \
  --checkpoint.restore_from="$T/tp2-cp2-ep2-fused-ckpt/global_step_5" \
  --checkpoint.checkpoint_dir="$T/resume-tp2-cp2-ep2-fused-ckpt"

ASCEND_RT_VISIBLE_DEVICES=0,1,3,4 FAMILY_TRACE_DIR="$T/ac-tp2-cp2-ep2-fused" \
$PY -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m tests.torch.attention.rfc793_trainer_validation "$T/tp2-cp2-ep2-fused.yaml" \
  --activation_checkpoint.mode=full \
  --checkpoint.checkpoint_dir="$T/ac-tp2-cp2-ep2-fused-ckpt"
```

其他拓扑替换文件名及卡数；8 个拓扑各 3 个模式的完整命令、物理卡、退出码保存在
`$T/commands.jsonl`。同目录日志与 trace 均按 `trace-/resume-/ac-` 加配置名命名。
运行与状态比较入口分离，避免把进程退出码 0 当成恢复精确一致：

```bash
$PY -m tests.torch.attention.compare_rfc793_training "$T" \
  --cases tp2-cp2-ep2-bf16 tp2-cp2-ep2-mxfp8 tp2-cp2-ep2-fused \
  --resume-cases tp2-cp2-ep2-fused --recompute-cases tp2-cp2-ep2-fused
```

层级设备测试入口（按空闲设备分开执行）：

```bash
ASCEND_RT_VISIBLE_DEVICES=3 $PY -m pytest -q tests/torch/attention/test_mxfp8_gqa_attention.py
ASCEND_RT_VISIBLE_DEVICES=4 $PY -m pytest -q tests/torch/attention/test_npu_attention_window.py
ASCEND_RT_VISIBLE_DEVICES=0,1,3,4 $PY -m pytest -q tests/torch/attention/test_gqa_parallel.py
ASCEND_RT_VISIBLE_DEVICES=0,1,3,4 $PY -m pytest -q tests/torch/expert_parallel/test_grouped_experts_ep_four.py
ASCEND_RT_VISIBLE_DEVICES=4,6 $PY -m pytest -q tests/torch/context_parallel/test_attention_family.py
ASCEND_RT_VISIBLE_DEVICES=4,6 $PY -m pytest -q tests/torch/attention/test_gqa_npu_cp.py
```

最后一条包含上述保留的 Qwen MXFP8 精度失败，不能预期为全绿。
R7 含两个 DSA skip；完整测试按本节报告的实际分段运行/重试记录解释。
完整 TP 梯度采集用 `rfc793_tp_gradient_validation collect CONFIG` 代替普通观察入口，
分别单卡和 TP2 跑首步，再执行：

```bash
$PY -m tests.torch.attention.rfc793_tp_gradient_validation compare \
  --reference-trace "$T/single-bf16-preclip-entry-trace" \
  --parallel-trace "$T/tp2-source-contract-bf16-trace" --tp-size 2 \
  --output /home/c00913822/hp-validation/tp-gradient-entry-report.json
```

统一 CPU 回归准确入口：

```bash
$PY -m pytest -q \
  tests/ut/components/modules/test_module_family.py \
  tests/ut/components/modules/test_family_defaults.py \
  tests/ut/components/modules/test_mxfp8_gqa_attention.py \
  tests/ut/components/modules/test_npu_attention_window.py \
  tests/ut/components/modules/test_kimi_delta_attention.py \
  tests/ut/components/quantization/test_mxfp8_memory.py \
  tests/ut/components/quantization/test_memory_contracts.py \
  tests/ut/components/quantization/test_saved_quantized.py \
  tests/ut/dual_mode_dtensor/test_module_replacement.py \
  tests/ut/dual_mode_dtensor/test_ep_moe_local_region.py \
  tests/ut/dual_mode_dtensor/test_precompiled_boundary.py \
  tests/ut/dual_mode_dtensor/test_plan_overrides.py \
  tests/ut/distributed/attention_family \
  tests/ut/distributed/tensor_parallel \
  tests/ut/distributed/context_parallel \
  tests/ut/trainer/test_config_overrides.py \
  tests/ut/trainer/test_replacement_overrides.py \
  tests/ut/trainer/test_tp_fsdp_setup.py \
  tests/ut/auto_models/checkpoint/test_source_layout_loading.py \
  tests/ut/auto_models/models/qwen3_5/test_qwen3_5_gdn_cp_contracts.py \
  tests/ut/auto_models/models/kimi_k3/test_kimi_k3_adapter_contracts.py \
  tests/ut/scripts/test_pylint_hyperparallel.py
```

FSDP2 附加配置从生成的121融合配置派生，仅把 `fsdp_config.dp_shard_size` 设为2并设置独立
checkpoint路径；也可在原配置后使用 `--fsdp_config.dp_shard_size=2` 和对应 checkpoint CLI 覆盖。
完整显式CP配置使用生成的 `tp1-cp2-ep2-fused-explicit.yaml`。其精确等价比较入口：

```bash
$PY -m tests.torch.attention.compare_rfc793_training "$T" \
  --cases tp1-cp2-ep2-fused tp1-cp2-ep2-fused-explicit \
  --equivalent-cases tp1-cp2-ep2-fused-explicit tp1-cp2-ep2-fused
```

## 静态质量检查

使用仓库 AutoGit 的原 `check` 命令和同一检查引擎，另对 tracked+untracked 全清单及真实
变更行做审计，覆盖77个文件（74个Python）。报告 `$V/STATIC_AUDIT.md`，基线对照
`$V/static-baseline-comparison.json`。新增 public 参数文档及新增复杂度问题已修复；
TP梯度比较器等价重构后，对已有真实梯度的42项比较仍通过。

**整体静态门禁没有全绿**：剩余9条pylint和7条lizard诊断在基线HEAD均已存在；
包括Torch-only目录仍被旧backend-import规则标记、外部函数静态推断和既有复杂函数。
完整测试规范检查还报60条UT缺arg_mark和60条UT缺三段式设备用例说明，
与仓库unit-test文档及既有unittest写法存在冲突。本轮没有关闭这些检查、调整阈值，
或为迎合设备测试规则批量给CPU用例添加ST标记。真实变更行的public文档检查通过。
版权检查修正为允许合法的既有版权归属及年份范围，仍检查Apache完整注释头；没有把Bytedance改署Huawei。

原AutoGit `check` 仅 `git add -u`，且检查前reset导致staged diff为空，不能覆盖新增文件并会
检查旧文件全部函数。因此保留原命令失败，另提供完整清单的同引擎结果，不把原命令当作充分覆盖。

设备矩阵在本轮工作树执行，随后形成独立提交。最终审查的文档补齐、等价函数拆分和
FSDP启用条件等价简化做了定向CPU回归；`is_causal=None` 修复新增真实SDPA的CPU对照，
没有为这些改动再次重复整套设备矩阵。原矩阵使用的缺省参数计算路径不变。

## 失败、修复和范围限制

- KDA chunk64 原先对未来 token 先 exp 再 mask，出现 inf×0 的 backward NaN。
  在 exp 前屏蔽非因果项后，CPU 独立 recurrent oracle 和真实 CP 前后向通过；未改有效因果项数学。
- 原 Trainer 观察器在首次 step 前调用 DCP optimizer.state_dict 导致隐式初始化 step，已改为只读状态。
  初始失败 trace 保留在 `$T/observer-failed-*`，正式恢复比较使用修复后的观察器。
- 曾在设备 2 的 checkpoint all-reduce 遇到 HCCL stream 资源耗尽；没有重置设备，改用空闲 4/6
  同代码重跑通过。R7 default-GDN、EP4 非融合也有 HCCL/SIGABRT 重试；原失败日志没有删除。
- 并行开发期间一次 builder/planner 参数签名不一致导致启动失败，修正后重新运行；
  首次 TP checkpoint shape/padding/梯度错误均保留诊断日志，最终证据来自修复后运行。
- DSA 缺 Omni 自定义算子及辅助 KL 的真实 CP 验收；GQA fusion=true 明确报能力错误，
  RFC 示例改为 false 的接受记录仍待 owner；EP8、PP、HiFloat8 不在本次通过范围。
- 训练是固定真实语料上的短程工程检查，不证明预训练收敛、生产模型质量或性能收益。
  逐步快照和摘要会同步设备；共享设备的耗时也不能用来宣称加速。

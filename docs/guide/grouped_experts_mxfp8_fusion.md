# GroupedExperts Family 设计对照与融合算子初步验证

日期：2026-09-24。当前工作基于功能提交 `d7e062413d53ca3d8686b29adf963ca7bce9cd41`，上游基线为 `36a9653b754966efab9cde8dee4068f09b861d11`。对照旧分支 `feat/mxfp8-swiglu-module` 的准确提交为 `25cbe09696240865cba478b9ba7e4c80c0e51d2f`。

本轮根据新增要求，扩展了上一阶段明确关闭的融合选项。融合具体指 **SwiGLU 与 MXFP8 双轴量化的前向及反向融合**。两次 GMM 仍独立执行，没有开发新 kernel，也没有将整个专家网络合成一个设备 kernel。

## Family 与继承的关系

两者解决不同问题，并且在当前方案中同时使用：

- `ModuleFamily` 决定配置最终实例化哪个类。注册项以 `GroupedExperts` 的精确类身份为键，不按名称猜测，不让所有子类自动继承注册资格。
- `MXFP8GroupedSwiGLU(GroupedExperts)` 决定低精实现复用哪些高性能能力。它继承参数组织、`forward`、`forward_expert_major` 和 `make_transforms`，仅替换本地专家计算。
- 本轮新增 `FusedMXFP8GroupedSwiGLU(MXFP8GroupedSwiGLU)`，继续复用上述契约，通过覆盖 `_local_compute` 切换融合实现。

因此，Family 是构建期的选择层；继承仍是模块内部的实现方式。

代码入口：

| 层次 | 代码 | 职责 |
| --- | --- | --- |
| Family 注册与选择 | `hyper_parallel/components/modules/family.py` | 消费两个选择开关，延迟导入最终类，严格校验构造参数 |
| 配置保真 | `hyper_parallel/trainer/config/replacement.py` | `ReplacementTarget` 保留用户指定的基线及开关，支持序列化和覆盖后重新选择 |
| replacement 接入 | `hyper_parallel/trainer/config/resolver.py`、`parallelism.py` | 进入现有 replacement 执行器，复用父子冲突和别名检查 |
| 模块继承 | `hyper_parallel/components/modules/mxfp8_grouped_swiglu.py` | 验证支持范围，保留路由和参数契约，分派本地计算 |
| EP grouped 选择 | `hyper_parallel/distributed/expert_parallel/experts.py` | 区分未指定、True、False；低精模块显式禁用 grouped 时拒绝构建 |
| 并行预检查 | `hyper_parallel/distributed/_builder/module_family.py` | 核验实际 mesh、EP 分片及最终覆盖配置，并提供 FQN 诊断 |

选择关系如下：

```text
用户始终指定 GroupedExperts
  use_mxfp8=false                           -> GroupedExperts
  use_mxfp8=true, use_fused_mxfp8=false      -> MXFP8GroupedSwiGLU
  use_mxfp8=true, use_fused_mxfp8=true       -> FusedMXFP8GroupedSwiGLU
```

`use_fused_mxfp8=true` 而 `use_mxfp8=false` 仍然报配置冲突。选择开关不会泄漏给基线构造器，未知参数不会被 `**kwargs` 吞掉。

## 与旧分支的区别

旧分支直接选择 `MXFP8GroupedSwiGLU`，通过其构造参数 `fused_swiglu_quant` 控制本地计算；当前配置保持 `GroupedExperts` 入口，由 Family 选择实现。两者都继承高性能类，也都保持 `[E,H,2I]`、`[E,I,H]` 两组持久权重及外层路由加权。

| 对比项 | 旧分支 | 当前实现 |
| --- | --- | --- |
| 模块组织 | 低精类继承 GroupedExperts | 保留该继承设计，新增 Family 选择层 |
| 配置入口 | 直接指定低精类及 `fused_swiglu_quant` | 指定 GroupedExperts 及 `use_mxfp8`、`use_fused_mxfp8` |
| 非融合计算 | 局部 functional 串联两次 GMM | 复用已合入 #862 修复的现有 GMM functional |
| 融合 autograd | 独立 SwiGLU Function 向相邻 GMM 传递 MXFP8Tensor carrier，反向也传递 carrier | 单个 Function 包含两次 GMM 和融合接口；外部输入、输出与梯度均为普通张量 |
| 反向保存 | 旧 GMM 将量化对象放在 ctx，反向后清除引用 | 非融合保留上游物理张量保存逻辑；融合通过 `save_for_backward` 保存物理 payload、scale、激活及 counts，重复反向时重新建立轻量包装 |
| EP 支持 | 有 grouped 本地计算协议 | 保留协议，并增加 grouped 三态选择和最终分片计划的定向预检查 |

没有整体拷贝旧分支的 GMM functional，也没有改动现有量化器对输入类型的契约。参考旧分支的两个 NPU 接口适配方法后，融合流程在新的独立文件中实现。

## 2026-10-09 完整 Trainer 实测补充

后续 A5 验收发现并修复了单卡原生权重加载后，GroupedExperts 转置替换未保留
已加载参数值的问题。修复后完成 EP=1/2 × BF16/非融合/融合六组各 10 步真实
Qwen3-MoE TextTrainer 训练、精确断点恢复，以及融合 full 重算对照。
环境来源、实际 CE 差异、完整命令、设备运行时异常和未验证范围见
[A5 真实训练验收记录](grouped_experts_mxfp8_a5_validation.md)。
下文历史数据仍对应原测试时间和源码版本。

## 本轮融合路径

配置示例：

```yaml
replace_module:
  _target_: hyper_parallel.components.modules.GroupedExperts
  use_mxfp8: true
  use_fused_mxfp8: true
```

新文件 `hyper_parallel/components/quantization/functional/mxfp8_fused_grouped_swiglu.py` 内的 `_FusedGroupedSwiGLU` 执行：

```text
前向：量化 X/W1 -> MXFP8 GMM1
                   -> npu_swiglu_mx_quant_with_dual_axis
                   -> MXFP8 GMM2 -> 普通输出张量

反向：量化 dY -> GMM2 的 dX/dW
                -> _npu_swiglu_backward_mx_quant_with_dual_axis
                -> GMM1 的 dX/dW -> 普通输入及权重梯度
```

融合 functional 直接接收规范持久权重布局，并返回对应布局的梯度；非融合路径仍通过可求导 transpose 适配已有 GMM functional。量化包装仅在融合 Function 内部参与算子调用，不穿过外部 autograd 边。保存张量 hook 和保留计算图后的重复反向已通过真实设备验证。

融合接口使用 **int64 累积 token 边界**，由本地 counts 构造，保证列向量化在专家边界重新分块。没有 token 时直接返回空输出及所需的零梯度，不调用不支持空输入的融合 kernel。模块构建时检查两个融合接口是否存在，缺少接口会报包含模块 FQN 的错误，不回退到非融合或 BF16。

仅支持先前的无 bias、标准 SwiGLU、三维 packed、TP=CP=PP=1、EP=1/2 范围。本次真实数值验收使用 BF16；FP16 未单独验证。仅支持一阶反向，未承诺高阶梯度。当前融合实现仍保存完整双轴量化操作数，冻结参数情况下的内存精简尚未专项优化。

## 测试结果

CPU：**50 项测试、137 项子测试通过**。新增真实配置与 replacement 的融合选择测试，以及缺失融合接口的构建错误测试；mock 仅隔离设备能力。

设备启动器：**4 项通过**，包括单卡（工作进程内分别验证融合和非融合）、非融合 EP=2、融合 EP=2、融合/非融合直接对照。EP 两种模式分别在新工作进程中运行，避免前一次进程组销毁后复用 mesh 缓存。

融合单卡规模 E=4、H=128、I=256、top-k=2，覆盖输出、dX、两个 dW、零 token 专家、空输入、分别冻结参数、冻结输入、重复反向与三步 SGD。观察到 **100 次真实 float8 GMM、14 次融合前向、23 次融合反向**。三步 loss 为 `0.22678428888320923`、`0.1692737340927124`、`0.13525497913360596`，两组权重均更新。

融合 EP=2 经真实配置、replacement、ShardingPlanner 和 apply_sharding_plan。局部权重为 `[2,128,512]`、`[2,256,128]`；正常路由每 rank 70 个 token，空 rank 场景分别为 140 和 0。输出、输入梯度、两个权重梯度和路由分数梯度，与控制 token 顺序、分组和归约方式的同模式参考完全一致。rank 0/1 分别执行 12/6 次 float8 GMM，以及 2/1 次融合前向和 2/1 次融合反向。

直接对照使用相同输入、权重、上游梯度及分组，counts 为 `[1,0,31,33]`、`[0,0,0,0]`、`[0,32,1,31]`。最大误差如下：

| 对象 | 融合与非融合的最大 NRMSE | 最大绝对误差 |
| --- | --- | --- |
| 输出 | 0.0109204 | 0.000610352 |
| 输入梯度 | 0.0113005 | 0.000793457 |
| Gate/up 权重梯度 | 0.0106026 | 0.0273438 |
| Down 权重梯度 | 0.0113383 | 0.0585938 |

沿用首次执行前固定的判定标准：对每个输出/梯度单独要求 RMS(error) ≤ 0.10 × RMS(reference) + 1e-6，同时 max-abs(error) ≤ 0.5 × max-abs(reference) + 1e-6。重复反向及 EP 同模式参考要求完全一致。没有根据结果放宽阈值。融合与非融合不要求逐位一致；融合改变了中间激活和导数的舍入路径。

小规模耗时对照：E=4、H=128、I=256，总 token 数 1024，每专家 256；测量本地专家完整前向、标量 loss 和反向，不含路由、EP 通信及优化器更新。预热 3 次、采样 10 次，每次计时前后同步 NPU。

| 路径 | 前向加反向耗时中位数 |
| --- | --- |
| 非融合 | 2.884560 ms |
| 融合 | 2.350395 ms |

该次测量耗时约降低 **18.5%**，约为 **1.23 倍**速度。它是共享设备环境中的小模型初测，包含 Python 调度和同步开销；不是正式性能基准，不能推断大模型训练收益。本次没有宣称完整模型收敛或全仓库回归通过。

## 环境、命令与证据

本次为 Ascend950PR，单卡使用设备 0，EP=2 使用设备 0/1；Python 3.10.21、PyTorch 2.9.0+cpu、CANN 9.2.0-beta.2。共享环境中的 torch-npu 已变为 **2.9.0.post7+gitddd16dd**，NumPy 1.26.4；因此融合与非融合均在本次同一环境中重跑，不跨版本比较耗时。

```bash
cd /home/c00913822/hyper-parallel
export HTTP_PROXY=http://127.0.0.1:7893
export HTTPS_PROXY=http://127.0.0.1:7893
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
source /home/zhaoyu65/cann-9.2.0/ascend-toolkit/set_env.sh
export PATH=/home/c00913822/hp-venv/bin:$PATH
export OMP_NUM_THREADS=2

python -m pytest -q \
  tests/ut/components/modules/test_module_family.py \
  tests/ut/components/quantization/test_mxfp8_memory.py \
  tests/ut/components/quantization/test_memory_contracts.py \
  tests/ut/components/quantization/test_saved_quantized.py \
  tests/ut/dual_mode_dtensor/test_module_replacement.py \
  tests/ut/dual_mode_dtensor/test_ep_moe_local_region.py

python -m pytest -q -s tests/torch/expert_parallel/test_grouped_experts_mxfp8.py
```

证据目录：`/home/c00913822/hp-validation/fused/`。最终结果见 `cpu-final.log`、`launchers-verified.log`、按测试名称和 rank 保存的 stdout/stderr、`metrics.json`。早期调试记录包含 int32 边界不满足融合接口要求、autograd ctx 保留属性名冲突、能力检查导致参考模型深拷贝失败，以及同进程重复初始化通信后的缓存问题；以上问题均已修正，未通过放宽数值阈值或替代设备算子规避。

本轮新增/修改的代码集中在 Family 选择、低精模块、NPU 融合适配、新融合 functional 和对应 CPU/设备测试。原有 MXFP8 GMM、Dense functional、Router、All-to-All 和模型输出组合未改动。

## 2026-09-28：统一 #862 保存逻辑及 EP=2 直接对照

`use_fused_mxfp8` 已通过真实配置解析选择融合子类；本次继续保留旧分支的融合语义，即 SwiGLU 与 MXFP8 双轴量化的前向、反向融合，而不恢复跨 autograd Function 的量化 carrier 传递。

本次将 `_saved_quantized.py` 中的物理张量打包和重建逻辑提取为共享的 `pack_quantized_operands`、`unpack_quantized_operands`。原有 `save_quantized_operands`、`restore_quantized_operands` 的调用签名、两个操作数的槽位布局和 None 占位保持不变；融合 Function 的四组操作数也调用同一套共享逻辑，并用 `ctx.save_for_backward(counts, projected, *buffers)` 保存。ctx 上只保留重建元数据，不保存量化包装本身。这个改动将上一版“遵循同一保存原则”的独立实现收敛为实际共享代码。

新增 `test_ep_fused_comparison`，在同一通信进程组内分别构建两个模型，仅融合开关不同：固定同一权重种子，断言 planner 分片后的权重完全一致，使用相同输入、路由、FP32 路由分数和上游梯度。两个模型均经过现有配置、replacement、planner 与 apply 入口；没有手工切分被测模型的权重。

正常路由与空接收 rank 两种场景均检查输出、输入梯度、两组局部专家权重梯度和路由分数梯度。同时在实际模型执行中开启 saved-tensor hooks，显式拒绝 QuantizedTensor 并复制物理张量后恢复；单卡对照继续覆盖保留计算图后的重复反向。

真实 EP=2 直接对照结果，取两个 rank、两种路由场景的最大值：

| 对象 | 最大 NRMSE | 最大绝对误差 |
| --- | --- | --- |
| 输出 | 0.0110621 | 0.000549316 |
| 输入梯度 | 0.0126678 | 0.000732422 |
| Gate/up 权重梯度 | 0.0127188 | 0.0390625 |
| Down 权重梯度 | 0.0107630 | 0.0234375 |
| 路由分数梯度 | 0.0120558 | 0.00610352 |

上述差异满足原先固定的 RMS 与最大绝对误差阈值，没有放宽。每个模式与其同顺序、同量化分组的参考仍完全一致；融合与非融合之间不要求逐位一致。空接收 rank 的局部权重梯度均严格为零。

实际调用计数：非融合两个 rank 的融合调用均为零；融合 rank 0 的前向/反向各 2 次，rank 1 各 1 次。两种模式 rank 0/1 均分别执行 12/6 次真实 float8 GMM，没有静默回退。

本次完整结果：CPU **50 项测试、137 项子测试通过**；真实设备 **5 个启动用例通过**，包括融合/非融合单卡、两种模式各自的 EP=2、单卡直接对照和新增 EP=2 直接对照。共享保存逻辑和融合 functional 的定向 Pylint 检查通过，`git diff --check` 和 AGENTS 目录检查通过；未运行全仓库 UT/ST。

沿用上文 CPU 与设备启动器命令即可复现。仅运行新增双卡对照：

```bash
python -m pytest -q -s \
  tests/torch/expert_parallel/test_grouped_experts_mxfp8.py::test_ep_fused_comparison
```

本次环境为 Python 3.10.21、PyTorch 2.9.0+cpu、torch-npu 2.9.0.post7+gitddd16dd，设备 Ascend950PR_9589，使用设备 0/1。本次证据独立保存在 `/home/c00913822/hp-validation/fused-20260928/`：`cpu.log`、`devices.log`、各 rank stdout/stderr、`metrics.json`、`environment.json` 和 `pylint.log`。

## 2026-09-28：BF16、非融合 MXFP8 与融合 MXFP8 的 loss 对照

新增真实单卡 `test_three_way_loss`，三路通过相同的配置解析和 replacement 入口构建。BF16 使用高性能 GroupedExperts；两条低精路径分别设置融合开关为 false/true。固定初始权重、输入及路由，E=4、H=128、I=256、top-k=2、40 个输入 token，路由权重为 0.25/0.75；三路初始权重严格相等。

loss 定义为 `output.float().square().sum()`，采用 SGD、lr=0.1，三路分别完成 3 次前向、反向、更新。表中记录每次更新之前的 loss，因此第 1 步比较相同权重下的数值差异，第 2/3 步还包含各自更新后权重轨迹的差异。这是小模型算术验证 loss，不是语言模型交叉熵或完整模型收敛指标。

| 步数 | BF16 | 非融合 MXFP8 | 融合 MXFP8 | 融合相对非融合的绝对相对差异 | 融合相对 BF16 的绝对相对差异 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.229481772 | 0.226671070 | 0.226784289 | 0.049949% | 1.175467% |
| 2 | 0.169729814 | 0.169359803 | 0.169273734 | 0.050820% | 0.268709% |
| 3 | 0.134877682 | 0.135451347 | 0.135254979 | 0.144973% | 0.279733% |

相对差异定义为 `abs(actual-reference)/abs(reference)`。三路 loss 均下降；融合与非融合最大相对差异约 0.145%，融合与 BF16 最大约 1.175%，非融合与 BF16 最大约 1.225%。这些是测量值，不是另外制定的 loss 容差。

每一步输出仍执行原有 RMS 与最大绝对误差检查，loss 及全部梯度要求有限。真实融合前向/反向各执行 3 次，两条低精路径均确认有实际 float8 GMM 调用。新增启动器 **1 passed**；与本轮此前通过的 5 个设备启动用例合计覆盖 6 个用例，没有将分批执行描述成一次全量运行。CPU 回归结果仍为 50 项测试、137 项子测试通过。

沿用上文环境准备后，仅运行三路 loss 对照：

```bash
python -m pytest -q -s \
  tests/torch/expert_parallel/test_grouped_experts_mxfp8.py::test_three_way_loss
```

证据位于 `/home/c00913822/hp-validation/loss-20260928/`：`launcher.log`、`rank0-stdout.log`、`rank0-stderr.log`、`loss-comparison.json`。本次融合及物理保存机制扩展将作为独立提交追加到 `feat/grouped-experts-family-mxfp8`，保留已推送提交的历史；网络操作使用 `http://127.0.0.1:7893`。

# GroupedExperts Module Family：非融合 MXFP8 首个最小闭环交付报告

## 范围与基线

首次实现及验证日期：2026-09-23；最新上游适配及回归日期：2026-09-24。

- GitHub 身份：`yooooo00`（Ewing）。已连接的 GitHub 资料显示名称为 Ewing；依赖提交的作者为 `Ewing <cyyooooo00@gmail.com>`。当前仓库使用相同的作者名称和邮箱。
- 上游：`mindspore-ai/hyper-parallel`，当前基线为 `master` SHA `36a9653b754966efab9cde8dee4068f09b861d11`；首次验证基线为 `051d821fbb8c873983a640ba5db356f8c52c4540`。
- 本地分支：`feat/grouped-experts-family-mxfp8`。
- 功能背景为 [Issue #793](https://github.com/mindspore-ai/hyper-parallel/issues/793)。附件中的完整设计仅作为背景，本阶段范围以用户本轮要求为准。
- 2026-09-23 首次检查时，[PR #862](https://github.com/mindspore-ai/hyper-parallel/pull/862) **仍处于开放状态，尚未合并**。已从其 head `39945019247562bf80fb0a56f8add5628390a2b0` 显式整合必要的 MXFP8 生命周期修复：保存物理操作数、MXFP8 Dense/GMM 的保存与恢复、空输入与冻结参数处理，以及 MXFP8 CPU 生命周期测试。保留 Dense 改动是为了覆盖该依赖的回归验证，本功能没有新增 Dense/MLP Family，也没有修改 HiFloat8 实现。2026-09-24 已迁移到包含 #862 的上游基线，相关 functional 和生命周期测试直接复用上游版本，不再作为本功能分支的额外改动。
- 模块组织参考：`yooooo00/hyper-parallel-fork` 的 `feat/mxfp8-swiglu-module`，已核验 SHA 为 `25cbe09696240865cba478b9ba7e4c80c0e51d2f`。仅参考子类组织方式，没有复制旧候选分支的 functional 文件或融合 carrier 路径。
- 所有 shell 网络操作均使用 `http://127.0.0.1:7893`，包括克隆、拉取、REST 请求和依赖安装；仓库还设置了本地 `http.proxy`。最初通过 GitHub 连接器查询身份发生在收到代理专项要求之前，后续查询均显式通过代理发出。

支持范围：三维 packed、无 bias 的标准 SwiGLU 专家；持久权重布局为 `gate_up_proj=[E,H,2I]`、`down_proj=[E,I,H]`；H/I 可被 32 整除；路由权重由外层在 GMM2 后应用；TP=CP=PP=1，EP=1 或 2，采用 production 并行执行模式。下述真实数值验证覆盖 BF16，以及使用现有 Qwen3-MoE EP recipe、接口与之匹配的小型模型。

本次不包含：GQA/CP、TP、PP、MLP Family、HiFloat8、融合 MXFP8、新 kernel、DTensor validate 模式执行、完整模型收敛、FSDP 训练或性能收益验证。

## 配置与行为

```yaml
plan_overrides:
  - match: "*.mlp.experts"
    module_type: transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeExperts
    replace_module:
      _target_: hyper_parallel.components.modules.GroupedExperts
      use_mxfp8: true
      use_fused_mxfp8: false

  - match: "*.mlp"
    when: ep
    region_dispatch: false
    local_compute_fn:
      _target_: hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel.qwen3moe_ep_compute_fn
      # 不指定 use_grouped_gemm 时，自动选择。
```

保留模型现有的 Router 和输出组合 recipe。通过 `use_mxfp8` 选择本功能，无须设置独立的旧版全局 `low_precision` 策略；该旧策略对并行组合的限制仍然保留。

| 配置 | 结果 |
| --- | --- |
| 不指定选择参数，或两者均为 false | 使用原有 GroupedExperts 高性能基线 |
| `use_mxfp8: true`，融合未指定或为 false | 使用 MXFP8GroupedSwiGLU |
| 未启用 MXFP8，却启用融合 | 配置冲突 |
| MXFP8 与融合均为 true | 明确报不支持 |
| 未知选择参数或构造器关键字 | 报错，不允许被 kwargs 吞掉 |
| grouped 模式未指定或为 None | 根据最终专家模块的协议自动选择 |
| grouped 模式为 true | 保留用户显式启用 grouped 的选择 |
| grouped 模式为 false，且使用 MXFP8 | 构建时报错，错误包含专家模块 FQN |
| grouped 模式为 false，且使用基线 | 使用现有逐专家路径，通过视图适配规范权重布局 |

Family 注册按基线类的精确身份匹配，不依赖类名或子类猜测。专用于 replacement 的解析机制在序列化、CLI 和 Python 覆盖配置时保留基线类及选择参数。构造参数按最终实现进行校验，选择参数不会传入基线构造器。替换匹配、父子冲突检查和别名处理仍由现有执行器完成，没有引入第二轮替换。

新的低精子类继承 `forward`、`forward_expert_major` 和 `make_transforms`，仅替换本地专家计算：

```text
experts(...) -> 继承的路由流程 / EP binder -> forward_expert_major
            -> MXFP8 GMM1 -> 普通 SiLU 与乘法 -> MXFP8 GMM2
            -> 现有外层路由加权与输出组合
```

持久权重参数仍然只有两个，通过可求导的转置视图适配现有 functional 的 `[E,out,in]` 布局。旧类 `quantization.modules.MXFP8GroupedExperts` 的导入路径、公开构造接口和布局保持不变。counts 表示每个**本地**专家收到的 token 数，允许为零；输出保持 expert-major 顺序。本地 MXFP8 计算不会应用可选的 scores。

构建预检查验证实际 dense/expert mesh 及包含用户覆盖项的最终计划。EP=2 要求两个规范布局权重都由 planner 在专家轴上应用 `Shard(0)`，并使用模型 recipe 的计算工厂。不支持的并行轴、执行模式、专家数和分片布局会在分片前报错，并包含模块 FQN。Router、All-to-All 和输出组合保持不变；本地计算仍通过 `experts.__call__`，因此保留模块 hook 语义。

## 改动文件

下表中的文件名相对于同一行给出的目录。

| 部分 | 目录 | 文件 |
| --- | --- | --- |
| Family 与模块 | `hyper_parallel/components/modules/` | `family.py`、`mxfp8_grouped_swiglu.py`、`grouped_experts.py`、`__init__.py` |
| 配置 | `hyper_parallel/trainer/config/` | `replacement.py`、`resolver.py`、`parallelism.py` |
| 替换诊断 | `hyper_parallel/models/` | `replacement.py` |
| EP 构建预检查 | `hyper_parallel/distributed/` | `_builder/module_family.py`、`apply.py` |
| EP 本地计算与 recipe | `hyper_parallel/distributed/expert_parallel/` | `experts.py`、`recipes.py` |
| 现有模型 recipe | `hyper_parallel/models/qwen3_moe/adapter/distributed/` | `expert_parallel.py` |
| 共享测试模型 | `tests/common/` | `grouped_experts_family.py` |
| CPU 模块测试 | `tests/ut/components/modules/` | `test_module_family.py` |
| CPU EP 回归测试 | `tests/ut/dual_mode_dtensor/` | `test_ep_moe_local_region.py` |
| 真实设备测试 | `tests/torch/expert_parallel/` | `test_grouped_experts_mxfp8.py`、`_test_grouped_experts_mxfp8.py` |
| 交付报告 | `docs/guide/` | `grouped_experts_mxfp8.md` |

基线模块将可选设备算子的解析延迟到实际计算时。用户显式选择基线逐专家计算时，EP 权重读取逻辑还会适配规范的三维布局。

## 环境与准确运行命令

| 组件 | 实际验证环境 |
| --- | --- |
| 操作系统与架构 | openEuler 24.03 LTS-SP4，aarch64 |
| Python | 3.10.21 |
| PyTorch | 2.9.0+cpu，由 torch-npu 提供 NPU 后端 |
| torch-npu | 2.9.0.post7+git7148a13 |
| CANN / 已安装数学算子 | 9.2.0-beta.2 |
| 驱动 | 25.1.rc1.b087 |
| 硬件 | Ascend950PR；可见设备共 8 张，测试使用设备 0 和 1 |
| Transformers / pytest | 5.12.0 / 9.1.1 |
| Omni | 未安装；本次非融合路径不需要 |

本地虚拟环境复用机器已安装的 PyTorch/NPU 软件栈；额外 Python 依赖和仓库的可编辑安装位于 `/home/c00913822/hp-venv`，没有修改被复用的原环境。

最终验证命令如下：

```bash
cd /home/c00913822/hyper-parallel
export HTTP_PROXY=http://127.0.0.1:7893
export HTTPS_PROXY=http://127.0.0.1:7893
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
source /home/zhaoyu65/cann-9.2.0/ascend-toolkit/set_env.sh
export PATH=/home/c00913822/hp-venv/bin:$PATH
export OMP_NUM_THREADS=2

python -m pip install --proxy http://127.0.0.1:7893 --no-build-isolation --no-deps -e .
python -c 'import hyper_parallel; print(hyper_parallel.__file__)'

python -m pytest -q \
  tests/ut/components/modules/test_module_family.py \
  tests/ut/components/quantization/test_mxfp8_memory.py \
  tests/ut/dual_mode_dtensor/test_module_replacement.py \
  tests/ut/dual_mode_dtensor/test_ep_moe_local_region.py

python -m pytest -q -s tests/torch/expert_parallel/test_grouped_experts_mxfp8.py
```

轻量启动器使用仓库的 `torchrun_case` 辅助函数，将各 rank 的输出记录到 `logs/tests/torch/expert_parallel/`。验证过程中直接执行工作进程所用的等价命令如下：

```bash
python -m pytest -q -s \
  tests/torch/expert_parallel/_test_grouped_experts_mxfp8.py::test_single_device

python -m torch.distributed.run --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=29793 -m pytest -q -s \
  tests/torch/expert_parallel/_test_grouped_experts_mxfp8.py::test_ep_two
```

## 测试结果与数值判定标准

- CPU：**32 项测试及 51 项子测试通过**。测试经过真实配置解析与 replacement 入口；mock 仅隔离设备算术、设备能力检查和进程组元数据。覆盖选择参数及覆盖配置、未知关键字、结构拒绝、权重双向转换、meta/别名行为、父子及重复替换冲突、grouped 三态选择、基线逐专家数值和定向构建预检查。另在全新解释器中验证了无法导入 NPU/Omni 时仍可构建基线。
- 单张 A5：**通过**，模型规模为 E=4、H=128、I=256、top-k=2。源模块与高性能 BF16 基线的输出满足一致性标准；MXFP8 的输出、dX 和 dW 满足下表阈值。counts 覆盖 0、1、31、32、33；测试包含本地空输入、完整路由空输入、分别冻结两个投影、冻结全部权重、冻结输入和重复反向。完成三步前向、反向及 SGD 更新，两个权重均发生更新。
- 单卡调用记录观察到 **100 次真实 float8 GMM 调用**，包含权重梯度 GMM。每次调用均委托实际 NPU 适配器执行，并断言两个操作数均为 float8；没有替换真实算术计算。
- EP=2：**两个 rank 均通过**，完整经过配置、replacement、`ShardingPlanner.plan`、`apply_sharding_plan` 和现有 Qwen3 recipe。验证了局部权重形状 `[2,128,512]` 与 `[2,256,128]`。正常路由时每个 rank 收到 70 个 token；空接收 rank 场景中，rank 0 收到 140 个，rank 1 收到 0 个。两个场景均触发模块 hook，包括未收到 token 的 rank。rank 0/1 分别执行 12/6 次真实 float8 GMM 调用。
- EP 输出、输入梯度、两个专家权重梯度以及路由分数梯度与参考结果**完全一致**。其中包含 FP32 路由分数，覆盖现有 dispatcher 将其转换为 BF16 时产生的舍入行为。

上游已有测试未为完整 MXFP8 专家计算链定义数值容差。在执行数值测试之前，已固定以下初步验证标准：

| 对照对象 | 判定标准与适用范围 |
| --- | --- |
| 源模块与高性能 BF16 基线 | 路由后的输出：`rtol=1e-2`、`atol=1e-4` |
| BF16 与非融合 MXFP8 | 分别检查输出、dX 和每个 dW：RMS(error) ≤ 0.10 × RMS(reference) + 1e-6；max-abs(error) ≤ 0.5 × max-abs(reference) + 1e-6 |
| 保留计算图后的重复反向 | 完全一致，`rtol=atol=0` |
| EP 与 token 顺序一致的 MXFP8 参考 | 输出、dX、两个 dW 和路由分数梯度均完全一致 |
| 检查点权重转换与空输入梯度 | 完全一致 / 严格为零 |

很小的绝对误差下限用于定义参考值接近零时的判定行为。这些标准仅用于算子计算链的初步验证，不代表模型质量或收敛目标。没有在查看结果后放宽任何阈值。

| 对象 | 实测最大 NRMSE | 实测最大绝对误差 |
| --- | --- | --- |
| 输出 | 0.0679362 | 0.00329590 |
| 输入梯度 | 0.0674490 | 0.00352287 |
| Gate/up 权重梯度 | 0.0680708 | 0.0859375 |
| Down 权重梯度 | 0.0671570 | 0.0742188 |

三步训练的 loss：`0.22667107`、`0.16935980`、`0.13545135`。

EP 参考实现复现了源 rank 拼接、目标排序、相同的本地 argsort、局部专家分组、逆置换，以及按发送方执行的 index-add 归约。因此，权重梯度量化看到的专家内分块也保持一致。只有参考实现手动切分完整权重；被测模型的局部参数完全由真实 planner/applier 产生。

## 首次验证的静态检查与留存证据（2026-09-23）

- `git diff --check`：通过。
- 新增 Python 文件：仓库 AutoGit 检查函数通过，包括 Pylint、复杂度、拼写、类型与文档约定及测试文档检查。按路径匹配的覆盖提示未识别这些集成测试；实际执行命令与结果已在上文记录。
- 对全部改动文件运行 `autogit.py check`：因已有文件中继承的检查问题，未全部通过。使用相同文件列表和配置检查干净的上游 worktree：基线 Pylint 问题数为 111，当前为 94，**没有新增问题**。已有问题包含历史文件头/插件策略、Torch 导入与推断警告、旧测试文档等；本次没有全局屏蔽这些检查。
- `python .agent/scripts/check_agents_catalog.py`：通过。
- 未运行全仓库 UT/ST。本次验收范围仅为选定的回归文件及新增单卡、EP=2 启动器。

本机证据保存在 `/home/c00913822/hp-validation/`：包含 `cpu.log`、`launchers.log`、各 rank 的 stdout/stderr、`metrics.json`、`environment.json`、AutoGit 报告以及基线/当前 Pylint JSON。

CPU mock 通过结果未计入真实硬件或分布式验证结果。

## 2026-09-24 上游同步与回归

通过 `http://127.0.0.1:7893` 代理重新拉取并核验，上游 `master` SHA 为 `36a9653b754966efab9cde8dee4068f09b861d11`。功能分支已迁移到该基线，包含首次验证之后的 10 个主线提交；#862 已由合并提交 `ba721027` 纳入上游。

迁移前保留本地备份分支 `backup/grouped-experts-mxfp8-before-upstream-20260924`。唯一文本冲突位于模块导出列表，解决时保留上游 `__all__` 初始化与动态扩展方式，只添加新的模块导出。上游 replacement 逐个安装、弱引用计划等生命周期改动保持完整；没有用旧文件覆盖。当前功能差异共 19 个文件，已合入上游的 #862 修复不再重复出现在差异中。

复用上文的代理、CANN、虚拟环境和 `OMP_NUM_THREADS=2` 设置，执行以下回归命令：

```bash
python -m pytest -q \
  tests/ut/components/modules/test_module_family.py \
  tests/ut/components/quantization/test_mxfp8_memory.py \
  tests/ut/components/quantization/test_memory_contracts.py \
  tests/ut/components/quantization/test_saved_quantized.py \
  tests/ut/dual_mode_dtensor/test_module_replacement.py \
  tests/ut/dual_mode_dtensor/test_ep_moe_local_region.py

python -m pytest -q -s tests/torch/expert_parallel/test_grouped_experts_mxfp8.py
```

CPU 回归通过：**49 项测试、138 项子测试**，包含上游新增的 replacement 生命周期测试和 #862 的完整 CPU 回归范围。

真实设备回归通过：**单卡及 EP=2 两个启动器均通过**。单卡仍记录到 100 次真实 float8 GMM 调用，三步 loss 与首次验证一致；EP 两个 rank 分别记录到 12/6 次调用，正常路由及空接收 rank 场景的输出和全部被测梯度均与参考完全一致。沿用原数值阈值，没有调整。

首次启动中，单卡工作进程曾在导入 NumPy 时因找不到 `libscipy_openblas64_-128b20d9.so` 而失败，未进入算子测试；同轮 EP=2 通过。随后确认共享环境的 NumPy 1.26.4、PyTorch 2.9.0+cpu 和当前仓库均可正常导入，在未修改代码、未更换算子、未修改环境配置或阈值的情况下重新运行两个启动器，结果为 2 passed。首次失败日志和重试成功日志均予以保留，不将该次导入失败记为数值验证通过。

本次 `git diff --check upstream/master` 与 AGENTS 目录检查通过。9 月 23 日的完整静态分析计数仅对应首次验证基线，本次没有重新执行整套静态分析或全仓库 UT/ST。

本次证据独立保存在 `/home/c00913822/hp-validation/upstream-20260924/`，包括 `cpu.log`、`launchers.log`、`launchers-retry.log` 和成功运行的各 rank stdout/stderr。推送目标为 `yooooo00/hyper-parallel-fork`：`master` 同步到上述上游 SHA，功能分支为 `feat/grouped-experts-family-mxfp8`。GitHub 凭据已核验属于 `yooooo00`（Ewing），网络请求均通过 7893 端口代理。

# A5 故障归因、修复与复验（2026-10-19）

本次针对首次主线迁移检查点的 HCCL/Trainer 故障追查。结果不是单一归因：
HCCL 初始化失败来自宿主机错误路径；通信恢复后发现的 HSDP 参数视图错误是纯主线已有缺陷。
两者均已修复。验证配置另有源样本被动态打包、实际更新数不足的问题，也已修正并加入完成步数断言。
本报告不宣称 QAT Family 已实现，block128 的独立问题仍保留在后续 QAT 工作中。

## 固定代码与环境

- 官方主线：`ce7d4b0051a6003de76c3401e4de4a6433a09497`。
- 本轮起点：`382c5e190c230942b611e767456f5a38adf94d11`。
- 修复代码：`561f2ba6b17ab449cb8e5272357e0b421daa324a`。
- 分支：`integration/module-family-mainline-20261018T194051Z-1071304`。
- 纯主线对照工作树：`/home/c00913822/hyper-parallel-upstream-ce7d4b00`，未修改主线实现。
- 独立对照 venv：`/home/c00913822/hp-upstream-control-venv`；迁移 venv：`/home/c00913822/hp-mainline-qat-venv`。
- 两者使用同一套已有 Python 3.10.21、torch 2.9.0+cpu、torch_npu 2.9.0.post7+gitddd16dd、CANN 9.2.0 beta.2；没有升级共享依赖。
- 所有证据在 `/home/c00913822/hp-validation/hccl-fix-20261019`（下称 `V`）。

## 1. HCCL 环境根因

`/etc/hccl_rootinfo.json` 应为可读取的 JSON 文件或不存在，现场却是空目录。
目录元数据记录创建/修改时间为北京时间 2026-10-17 14:46:53。
`strace` 显示 HCCL 打开该路径；详细 plog 随后报 JSON 解析失败、`RootInfoDetect` 失败。

对照没有改变 Torch/CANN、输入或实现：

| 对照 | 结果 |
| --- | --- |
| 不导入 HyperParallel 的最小 HCCL 程序，原宿主机路径 | JSON 解析失败/初始化阻塞 |
| 私有 mount namespace 仅隐藏该目录，物理卡0/1 | 两 rank 同步取回正确 all_reduce 结果3.0 |
| 纯官方主线 Trainer 的 all_reduce 封装，原路径 | 同样失败，源码导入路径记录在日志 |
| 纯官方主线同一封装，私有 namespace 隐藏目录 | 两 rank 均通过 |
| 宿主机修复后，独立单/双 rank 自检 | 均通过，结果分别1.0/3.0 |

证据：`open.trace`、`plog/debug/plog/`、`control-debug.log`、`private-fix-01.log`、
`upstream-before.log`、`upstream-before.plog`、`upstream-private-after.log`、`probe-final-single.log`、`probe-final-double.log`。
阻塞的原对照只终止了本次创建的进程，终止记录不冒充正常完成。

修复时核验了路径为非符号链接的空目录，原样改名备份为：
`/etc/hccl_rootinfo.json.invalid-directory-20261019`。
没有伪造拓扑 JSON、重启驱动或重置设备。操作时间、inode 和回退命令见 `environment-repair.json`。

发现运行中的 `crm_qwen3.8_27b` 容器绑定了该宿主机路径。此配置有再次创建异常目录的风险；
现有证据不足以断言哪个进程最初创建了该目录。没有修改或重启他人的服务。
后续创建容器时，应仅在源路径是普通文件时添加该挂载，并优先用缺失源路径会报错的 `--mount`：

```bash
mount_args=()
if test -f /etc/hccl_rootinfo.json; then
  mount_args+=(--mount type=bind,src=/etc/hccl_rootinfo.json,dst=/etc/hccl_rootinfo.json,readonly)
fi
# 将 "${mount_args[@]}" 加入既有 docker run 参数；不要无条件 -v 挂载缺失文件。
```

2/3号卡的一个隔离试验另遇到3号卡驱动流创建错误 `halSqCqAllocate/507899`；未将它误计为 HCCL JSON 故障。
正式 EP 复验使用已验证可用的0/1号卡，单卡 Trainer 使用5号卡，未占用已有服务的4号卡。

## 2. 纯主线已有的 HSDP 代码缺陷

HCCL 修复后，EP=2 BF16/MXFP8/融合 Trainer 均在首次 unshard、尚未进入专家计算时失败：
`A view was created in no_grad mode ... modified inplace with grad mode enabled`。

用纯主线 `TextTrainer` 和不含 Family selector 的 BF16 配置复现相同错误。
进一步以 CPU、真实 HSDPParamV2 和真实 autograd 缩小复现，不需要 NPU 或真实通信：
meta 参数物化为可训练本地 Tensor，在 no_grad 下 reset，参数同步更新共享存储，然后 unshard。
纯主线的回归测试在 trainable=True 子项失败。

根因在 `HSDPParamV2._refresh_sharded_local_tensor()`：
它先创建通信展平视图，随后把该视图的本地基 Tensor 设为可训练。
通信缓冲区因此继承了参数的 autograd 视图关系；同步写入后，构造 unsharded 参数触发版本检查。

修复对两条重建分支的展平通信视图显式 `detach()`。存储仍共享，不复制第二份持久权重，
不取消参数梯度、不跳过 PyTorch 版本检查、不重新创建优化器持有的 Parameter。
回归覆盖冻结/可训练、共享存储、同步后首次 unshard、真实梯度，以及更新后稳定 Parameter 的值刷新。

证据：`upstream-trainer-before.log`、`hsdp-cpu-before.log`、`hsdp-regression-upstream.log`、`hsdp-cpu-after.log`。

## 3. 验证配置修复

迁移验证生成器使用主线 TokenBatchLoader 后，默认64 token预算把20条源样本打包为10个微批次，
每步消费2个微批次，因此原来设置 train_iters=10 的配置实际上只执行5次更新就耗尽数据。
训练框架对动态迭代器计算出一轮，日志正常退出，不能据此声称完成10步。

验证生成器现设置 `token_budget=1`，使用 TokenBatchLoader 已有的“单条超过预算时单独输出”契约，
使每条源样本对应一个微批次。保留真实数据管线、模型、权重和优化器。
ObservedTextTrainer 在返回前检查实际 global_step 必须等于请求步数，避免再次假通过。
本次使用新目录 `trainer-fixed`，没有混入旧打包方式下的5步轨迹。
数据仍是固定合成文本和随机初始化小型原生 Qwen3-MoE，不是预训练模型或标准数据集。

## 4. 最终验证

| 项目 | 结果 |
| --- | --- |
| CPU：原迁移相关集合 + fully_shard 回归 | 505 passed，752 subtests passed |
| A5 原完整入口，宿主机修复后 | 8 passed |
| A5 原完整入口，HSDP 修复后再次运行 | 8 passed，0 failed，0 skipped（128.83秒） |
| EP1/EP2 × BF16/MXFP8/融合 | 六路各10个真实 optimizer step，检查点5/10保存成功 |
| 六路新进程从第5步恢复 | 第6–10步 loss、梯度范数、权重、批次、RNG 逐项完全一致 |
| 融合 EP1/EP2 full 非重入重算 | 第1–10步 loss、梯度范数、权重、批次、RNG 完全一致；融合前向调用数增加 |
| 低精实际调用 | 观察器确认 MXFP8 GMM；融合模式确认融合前向/反向调用，没有静默 BF16 fallback |
| 静态检查 | 未全部通过：HSDP 文件已有文档/类型/复杂度与测试覆盖识别项；本次两个新增 Python 文件的 pylint 通过 |

比较器对同拓扑恢复/重算的权重使用 `rtol=0, atol=0`，其它字段相等检查；原 A5 数值阈值不变。
跨精度训练 loss 仅如实报告，不在观察结果后追加或放宽验收阈值：

| 对比 | EP1 最大绝对 loss 差 | EP2 最大绝对 loss 差 |
| --- | --- | --- |
| 融合 vs 非融合 MXFP8 | 0.0005807877 | 0.0013520718 |
| 融合 vs BF16 | 0.0007977486 | 0.0014393330 |
| 非融合 MXFP8 vs BF16 | 0.0006766319 | 0.0011827946 |

详细轨迹、相对差与完整比较结果见 `training-comparison.json`。
没有用跨 EP 拓扑的量化数值差判断通信错误。

## 5. 可执行复验命令

```bash
cd /home/c00913822/hyper-parallel-family-mainline-20261018T194051Z-1071304
source /home/zhaoyu65/cann-9.2.0/ascend-toolkit/set_env.sh
export OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120
PY=/home/c00913822/hp-mainline-qat-venv/bin/python
V=/home/c00913822/hp-validation/hccl-fix-20261019

ASCEND_RT_VISIBLE_DEVICES=0,1 $PY -m torch.distributed.run \
  --standalone --nproc-per-node=2 -m examples.training_demo.check_a5_communication

$PY -m pytest -q tests/ut/core/fully_shard
ASCEND_RT_VISIBLE_DEVICES=0,1 $PY -m pytest -q -s \
  tests/torch/expert_parallel/test_grouped_experts_mxfp8.py

# 正式复跑请使用新目录，观察器会拒绝混入已有 JSONL。
$PY -m examples.training_demo.prepare_family_validation "$V/trainer-rerun"
ASCEND_RT_VISIBLE_DEVICES=0,1 FAMILY_TRACE_DIR="$V/trainer-rerun/trace-ep2-fused" \
  $PY -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m tests.torch.expert_parallel.family_trainer_validation "$V/trainer-rerun/ep2-fused.yaml"

# 检查本轮已经完成的六路训练、六路恢复和两路重算。
$PY -m tests.torch.expert_parallel.compare_family_training \
  "$V/trainer-fixed" --resume --activation-checkpoint
```

恢复配置使用 `checkpoint.restore_from=<相应目录>/global_step_5`，关闭恢复运行的保存；
重算配置使用 `activation_checkpoint.mode=full`。实际14份配置、所有日志和每 rank 轨迹均在 `trainer-fixed/`。

## 6. 尚未完成的工作及边界

- QAT Family 尚未实现；本次没有增加 selector、低精 Family 类或第二阶段并行能力。
- native W4A8 block128 的 FP32 per-token scale 错误、fake W4A8 block128 的 scale/weight 转置契约错误，
  在纯主线单次前向/反向中分别复现：`upstream-native128.log`、`upstream-fake128.log`。
  它们不由 Family 或本次 HSDP 修复引入，也没有通过本次环境修复解决；不能宣称五种 QAT 配置已可用。
- 前一检查点的完整迁移静态检查尚需收敛；本轮功能复验通过不能代替完整 C0 门禁，也不自动授权提前开始 QAT 阶段。
- AutoGit 初次因 PATH 未包含已有工具而尝试下载 pylint，已中止；重新配置本地工具 PATH 并设置 PIP_NO_INDEX=1。
  后续远程 Git 操作显式使用7893代理，没有升级共享依赖。

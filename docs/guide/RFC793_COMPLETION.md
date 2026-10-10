# RFC #793 实现与验收清单

本清单对应本轮 A5 任务书的 R1–R10。实现契约见
[Module Family 设计](module_family_implementation.md)，实际环境、运行命令、
原始日志、数值指标和失败重试见 [A5 验证报告](rfc793_a5_validation.md)。

**RFC 尚未关闭。** 公共机制和各协议的代码入口已补齐，但 DSA 的真实稀疏算子验收、
GQA 融合示例的 owner 接受记录，以及一项 GQA MXFP8 对 BF16 的梯度精度检查仍未通过。
“有实现”“有可执行测试”和“本轮测试通过”分别记录，不能互相替代。

## 提交基线

本轮从 `feat/grouped-experts-family-mxfp8` 的
`a3051b75250a87ae09a7c7c300b4fa63711ccce2` 继续开发。
本轮新增实现与测试按以下独立提交登记；文档提交只补充最终证据，不改变运行代码。

| 提交 | 内容 |
| --- | --- |
| `d246cbb75cebce83b68d5e102209a69c0409f2fe` | R1–R8：共用 Family/默认值、GQA 低精与 CP、TP/EP 能力、R7 adapter，以及相应 CPU/层级用例 |
| `80e7f0febaa3ee2eead950d86433052572ff16d3` | R5/R9：TP checkpoint 源布局、分片 padding 配套加载与 Trainer 梯度同步 |
| `55e47cda56646d7ba473ea8c3c73174bd272a48d` | 静态检查：保留版权归属、Apache 注释头和年份范围校验 |
| `c38d9d416821af026759cd5746dd8cce12937cd4` | R3–R6/R9/R10：A5 层级/EP4、真实 Trainer、恢复/重算比较器及配置生成器 |

## R1–R10 对照

表中测试名均指向真实代码入口；设备测试入口存在不代表已经运行。

| 条目 | 实现入口 | CPU 与设备验收入口 | 本轮状态及尚需证据 |
| --- | --- | --- | --- |
| R1：共用 Family 注册、选择、校验 | [family.py](../../hyper_parallel/components/modules/family.py)、[replacement.py](../../hyper_parallel/trainer/config/replacement.py)、[Family planner 校验](../../hyper_parallel/distributed/_builder/module_family.py) | [ModuleFamilyTests](../../tests/ut/components/modules/test_module_family.py)：选择/冲突/未知参数/父子重复替换；[FamilyDefaultsTests](../../tests/ut/components/modules/test_family_defaults.py)：精确身份、缺失变体、往返 | 六个 Family 共用注册、最终实例元数据及校验器分派。已通过本轮统一 CPU 回归。 |
| R2：默认 adapter、显式覆盖 | [planner.py](../../hyper_parallel/distributed/_builder/planner.py)、[rule_resolver.py](../../hyper_parallel/distributed/_builder/rule_resolver.py)、[forward_rewriter.py](../../hyper_parallel/distributed/_builder/forward_rewriter.py)、[plan.py](../../hyper_parallel/distributed/plan.py) | `FamilyDefaultsTests.test_default_and_explicit_priority`、`test_explicit_disable_is_not_overwritten`、`test_adapter_install_once_and_contract_guards`；[其他 Family 默认覆盖测试](../../tests/ut/distributed/attention_family/test_state.py) | 通过原 planner 合并和原 rewriter 安装，摘要记录来源；默认/显式、禁用及重复安装有 CPU 覆盖。设备层默认/显式对照由各 Family 独立验收。 |
| R3：GQA MXFP8 投影及权重契约 | [mxfp8_gqa_attention.py](../../hyper_parallel/components/modules/mxfp8_gqa_attention.py)、[gqa_attention.py](../../hyper_parallel/components/modules/gqa_attention.py)、[mxfp8_linear.py](../../hyper_parallel/components/quantization/modules/mxfp8_linear.py) | [GQA CPU 测试](../../tests/ut/components/modules/test_mxfp8_gqa_attention.py)：loaded/meta、逆转换、hooks、冻结、空输入；[单卡设备入口](../../tests/torch/attention/test_mxfp8_gqa_attention.py)：`test_gqa_projections` | Llama/Qwen3-MoE 共用继承式变体；当前低精声明为 BF16。已有单卡前后向及生命周期记录。native→baseline 与 baseline→MXFP8 使用独立比较口径，原 native BF16 逐元素失败不能改写为通过。 |
| R4：公共 GQA CP 协议 | [attention_interface.py](../../hyper_parallel/distributed/context_parallel/attention_interface.py)、[NPU Attention 接口](../../hyper_parallel/components/functional/npu_fusion_attention.py) | `FamilyDefaultsTests.test_global_masks_and_causal_offsets`、`test_invalid_mask_and_hidden_state_dimensions`、`test_all_packed_length_aliases_fail_before_communication`；[GQA 并行入口](../../tests/torch/attention/test_gqa_parallel.py)：`test_gqa_parallel`、`test_gqa_cp_masks` | 等长连续分片、全局 mask、K/V 可导 gather、明确拒绝 cache/packed/滑窗；两个 GQA 模型的层级 CP 有运行记录。完整默认 NPU kernel Trainer 单独验收，不能用 eager 层级通过替代。 |
| R5：TP 与 TP/CP/EP 组合 | [head_count.py](../../hyper_parallel/distributed/tensor_parallel/head_count.py)、[precompiled_boundary.py](../../hyper_parallel/distributed/_builder/precompiled_boundary.py)、[tp_region_gradients.py](../../hyper_parallel/distributed/_builder/tp_region_gradients.py)、[Family 布局校验](../../hyper_parallel/distributed/_builder/module_family.py) | [TP 边界梯度 CPU 测试](../../tests/ut/distributed/tensor_parallel/test_tp_region_gradients.py)；GQA 并行入口的 `tp`/`cp`/`tp_cp`、`test_gqa_tp_bias`、`test_gqa_tp_validate`；混合 Trainer 入口见 R9 | 局部对齐及参数角色检查已实现；TP 复制输入/输出的 adjoint 只对声明该契约的 Family 生效。层级 TP/CP 及完整 8 拓扑×3 精度训练通过；单卡→TP2 的 42 项完整模型梯度分片比较通过。FSDP2 构建、训练及恢复/重算已有通过记录，详见报告。 |
| R6：Experts 统一 EP 与组合检查 | [grouped_experts.py](../../hyper_parallel/components/modules/grouped_experts.py)、[mxfp8_grouped_swiglu.py](../../hyper_parallel/components/modules/mxfp8_grouped_swiglu.py)、[Family 布局校验](../../hyper_parallel/distributed/_builder/module_family.py)、[Qwen3-MoE EP recipe](../../hyper_parallel/models/qwen3_moe/adapter/distributed/expert_parallel.py) | `ModuleFamilyTests.test_grouped_tristate_and_module_call`、`test_grouped_binder_supported_expert_sizes`、`test_plan_preflight`；[原单卡/EP2 设备入口](../../tests/torch/expert_parallel/test_grouped_experts_mxfp8.py)、[EP4 入口](../../tests/torch/expert_parallel/test_grouped_experts_ep_four.py) | 保留原路由、All-to-All、combine 和 `experts(...)`；代码范围 EP=1/2/4，按最终布局定向放开 TP/CP。EP4 三变体、原 EP2 和组合训练已运行通过；EP8 未开放，不记为已通过。 |
| R7：四类 Attention adapter | [attention_family 目录](../../hyper_parallel/distributed/attention_family)、[MLA](../../hyper_parallel/components/modules/mla_attention.py)、[DSA](../../hyper_parallel/components/modules/dsa_attention.py)、[KDA](../../hyper_parallel/components/modules/kimi_delta_attention.py) | [MLA CPU](../../tests/ut/distributed/attention_family/test_mla.py)、[DSA CPU](../../tests/ut/distributed/attention_family/test_dsa.py)、[state CPU](../../tests/ut/distributed/attention_family/test_state.py)；[设备入口](../../tests/torch/context_parallel/test_attention_family.py)：`test_mla_family_cp`、`test_dsa_family_cp`、`test_state_family_cp`，各含 default/explicit | MLA/GDN/KDA 已有真实 CP1/2 前后向记录；DSA 缺 Omni 自定义算子包，真实测试明确 skip。DSA indexer 辅助 KL 的 CP 全局归一化也未验收；R7 未关闭。 |
| R8：按 Family 解析融合语义 | [Family 变体登记](../../hyper_parallel/components/modules/family.py)、[融合 Experts](../../hyper_parallel/components/modules/mxfp8_grouped_swiglu.py)、[物理 tensor 保存的融合 Function](../../hyper_parallel/components/quantization/functional/mxfp8_fused_grouped_swiglu.py) | `ModuleFamilyTests.test_fused_selection_and_capability`；`FamilyDefaultsTests.test_missing_variant_and_round_trip`；原设备入口的 `test_fused_comparison`、`test_ep_fused_comparison`、`test_fused_lifecycle_and_boundary` | Experts 开关选择真实融合前后向。GQA 不登记不存在的融合变体，`true` 报能力错误。RFC GQA 示例修订为 `false` 的 owner 接受记录尚未收到；R8 保持设计差异，不能称原示例可运行。 |
| R9：训练、恢复、重算、回归 | [新 Trainer 观察入口](../../tests/torch/attention/rfc793_trainer_validation.py)、[训练比较器](../../tests/torch/attention/compare_rfc793_training.py)、[已有 Experts Trainer](../../tests/torch/expert_parallel/family_trainer_validation.py) | 单卡与最低拓扑矩阵、同拓扑新进程恢复、full 重算及物理保存/offload；本轮重跑既有 CPU/NPU/六路 Trainer 入口 | 8 拓扑×3 模式完成真实语料训练（前7种各10步，222各50步）；另有 FSDP2 和显式 CP 配置。原8个NPU入口和六路Trainer重跑通过。11项恢复、5项full重算及default/explicit完整配置的逐步参数/状态精确比较通过，见报告。保留 Qwen MXFP8 q_norm 梯度 NRMSE .11585>.10，以及 native BF16 严格逐元素差异；数值验收未全部关闭。 |
| R10：公共配置、示例、文档 | [parser.py](../../hyper_parallel/trainer/config/parser.py)、[resolver.py](../../hyper_parallel/trainer/config/resolver.py)、[完整配置生成器](../../examples/training_demo/prepare_rfc793_validation.py)、[实现设计](module_family_implementation.md) | [CLI replacement CPU 测试](../../tests/ut/trainer/test_replacement_overrides.py)、[通用 CLI 覆盖测试](../../tests/ut/trainer/test_config_overrides.py)；生成器的真实配置解析/往返 | 列表索引覆盖、整批选择器校验、配置往返及完整本地资产生成已实现。生成的 36 份拓扑/default/explicit 配置均通过真实解析和往返；默认/显式完整配置均已运行。提交号及设备结果在本清单和独立报告登记，设计文档不充当运行通过证明。 |

## 明确的设计差异和未关闭项

1. **GQA 融合示例。** RFC 原示例的 `use_fused_mxfp8=true` 没有本轮对应的投影融合算子。
   本实现使用 `use_mxfp8=true, use_fused_mxfp8=false`；对 `true` 明确报错。
   这不是把 packed QKV 或普通融合 Attention 重新命名为低精融合。已提出示例修订，
   owner 尚未答复，等待不能视为接受。
2. **DSA。** 需要 `omni_training_custom_ops` 注册的稀疏 Attention、indexer 和 indexer 梯度算子。
   CPU 仅隔离可选算子的 import，不产生真实数值通过证据。安装依赖后仍须运行 default/explicit CP2、
   输出/dX/dW 和 indexer 辅助 KL 的全局归一化对照；R7 保持未完成。
3. **低精数值。** 全局96 token 的 Qwen3-MoE MXFP8 CP2 用例，q_norm.weight 梯度
   NRMSE=.1158468859，超过预设 .10；没有提高阈值。同 MXFP8 量化顺序的无通信 oracle
   与 CP 精确一致，定位为本用例的低精误差，不能用该通信对照或完整 Trainer 成功关闭精度失败。
4. **验证范围。** GQA MXFP8 当前仅 BF16，其他四类 Attention 仅声明各自 baseline CP 协议；
   TP/PP 扩展、EP8、新 FP8 Attention kernel、低精通信和拓扑热迁移不计为已支持。
   原 native BF16 与 packed baseline 的不同归约/算子路径产生的误差须单独披露，不能放宽旧阈值隐藏。

## 观察与恢复比较的口径

训练观察器委托原配置解析、replacement、planner、apply、Trainer 和真实设备算子。
它按模块记录 forward 次数、非空输入次数、可训练参数梯度和每步 FP8/融合调用增量，
不会因另一个模块正常工作而让一个被跳过的低精模块通过。

optimizer 摘要读取底层已存在的 `state` 和 `param_groups`，以稳定 FQN/顺序映射参数，
不调用可能初始化状态或执行优化器更新的 DCP `state_dict`。
缺省的 group-level `step` 与恢复补入的 `None` 视作同一占位表示；
实际的整数 step、每参数 step、moment tensor、主权重和 scheduler 状态均保留在摘要中。
摘要包含 tensor dtype、shape 和本地物理字节，不包含进程内对象地址。

逐步快照和摘要会同步设备；该入口用于正确性验收，不用于性能结论。

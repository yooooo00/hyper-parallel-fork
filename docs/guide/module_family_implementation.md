# Module Family：实现选择与并行适配

本文描述实现契约。RFC 条目状态见 [完成清单](RFC793_COMPLETION.md)，
设备、命令、误差和失败记录见 [A5 验证报告](rfc793_a5_validation.md)。

## 用户入口

用户指定高性能 baseline 的 replacement target，通过精度开关选择最终实现。
Family 选择不替代模型 recipe：模型继续提供 Router、EP 本地计算入口及输出组合。

```yaml
accelerator:
  tp_size: 2
  cp_size: 2
  ep_size: 2
  pp_size: 1
  sequence_parallel: false
  loss_parallel: false

plan_overrides:
  - match: "*.self_attn"
    module_type: transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeAttention
    replace_module:
      _target_: hyper_parallel.components.modules.GQAAttention
      use_mxfp8: true
      use_fused_mxfp8: false
  - match: "*.mlp.experts"
    module_type: transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeExperts
    replace_module:
      _target_: hyper_parallel.components.modules.GroupedExperts
      use_mxfp8: true
      use_fused_mxfp8: true
  - match: "*.mlp"
    when: ep
    region_dispatch: false
    local_compute_fn:
      _target_: hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel.qwen3moe_ep_compute_fn
```

完整配置由 `examples/training_demo/prepare_rfc793_validation.py` 生成，包括模型、
本地 tokenizer/数据、AdamW、checkpoint 和 FSDP 设置。上面的片段省略了这些必需字段。

CLI 列表索引覆盖从 `parse_training_args()` 进入同一解析流程：

```bash
python -m examples.training_demo.train_text config.yaml \
  --plan_overrides.1.replace_module.use_mxfp8=true \
  --plan_overrides.1.replace_module.use_fused_mxfp8=true
```

索引必须指向现有项，不允许负索引或隐式追加。整批覆盖完成后重新构建 replacement，
两个开关的书写顺序不改变最终选择。布尔值、未知参数和缺少的列表路径均作校验。

## 注册和构建链路

`components/modules/family.py` 中的 `ModuleFamily` 声明：

| 字段 | 用途 |
| --- | --- |
| `baseline` | 精确匹配的构造器身份；子类须另行登记 |
| `variants` | 明确支持的低精实现导入路径 |
| `parallel_defaults` | 按并行轴登记默认 adapter |
| `plan_validator` | 最终布局和组合约束的校验器 |
| `parameter_layout` | 构建摘要中的持久参数契约 |
| `tp_replicated_gradient_contract` | 对 TP 复制输入/输出使用特定反向归约规则 |

模块类型的字符串名称相同不等于注册身份相同。查询最终模块时优先读取 replacement
写入的 `FamilySelection`，也允许识别精确注册的 baseline/变体类型。
注册表使用导入路径延迟加载可选设备实现；仅解析 baseline 配置不要求导入所有低精算子。

构建顺序为：配置解析 → 消费选择参数 → replacement 执行 → planner 推导及显式 contract 覆盖/插入 →
补齐未声明的 Family adapter → 最终能力校验 → 原 forward rewriter 安装。
`ShardingPlan.explain()` 显示 FQN、Family、变体、实现、adapter 来源、并行度及参数布局。

选择器必须为 bool；`use_fused_mxfp8=true` 要求 `use_mxfp8=true`。
baseline 构造器不会收到这些选择器。未知参数按最终构造器的具名参数检查，
不能因构造器存在 `**kwargs` 而静默接受。保存配置保留 baseline target 和选择参数，
再次解析时重新选型。缺少变体返回包含 Family、选择参数和模块路径的能力错误。

## 默认 adapter 和覆盖

优先级为显式设置、Family 默认值、通用默认值。

| `inner_wrapper` 输入 | 合并语义 |
| --- | --- |
| 未指定、`null`、`auto` | 继承已推导的默认值 |
| `none`、`false` | 明确禁用，并清除相应 inner target/output 契约 |
| callable 或 `_target_` 配置 | 使用显式 adapter |

如果 CP 必需的 adapter 被禁用，构建会报包含 FQN 的错误，不恢复默认值。
`derive=False` 的完整显式 contract 也使用同一规则；先插入边界，再补齐默认 adapter，
不会因为模板推导关闭而提前拒绝合法边界。输入的 override 对象保持不变，可重复生成 plan。
已有模型显式 wrapper 保留优先级。wrapper 通过 `_ForwardRewriteRequest` 一次性安装
forward 和 companion attributes，避免在构造器中零散改写 forward。
GQA/MLA 公共接口 adapter 对同一 mesh 重复安装不再套一层；换 mesh 须重新构建模块。
state adapter 对重复安装明确报错。

## GroupedExperts

`MXFP8GroupedSwiGLU` 继承 `GroupedExperts`，融合类继承这一低精变体。
三者复用参数组织、`make_transforms()` 和路由流程，仅替换本地专家计算。

| 持久参数 | 布局 |
| --- | --- |
| `gate_up_proj` | `[E, H, 2I]` |
| `down_proj` | `[E, I, H]` |

functional 边界的转置可求导，不创建第二套持久权重。
已加载源权重直接复制为目标布局；meta 路径使用可逆 checkpoint 转换，保留 dtype 和冻结状态。
旧低精类的公开布局与导入路径保持兼容。

低精范围为三维 packed、标准 SwiGLU、无专家 bias，且 GMM2 后由外层应用路由权重。
`forward_expert_major(x, num_tokens_per_expert, scores=None)` 的第二个参数
（协议中简称 counts）表示每个本地专家的 token 数，
返回顺序保持 expert-major。路由权重沿原位置和 dtype 应用一次。

EP 路径继续调用 `experts(...)`，保留模块 hooks。未指定 grouped 模式时按最终模块能力选择；
显式 `False` 对仅支持 grouped 的低精变体报错。支持的 EP 度为 1、2、4，并须整除专家数。
EP>1 时两项权重必须由 planner 在 EP 轴 `Shard(0)`，且有模型 recipe 的本地计算函数。
不允许对专家矩阵附加 dense TP 切分；TP 开启而 EP 关闭的示例显式保持专家参数复制。

融合开关选择实际 SwiGLU/量化融合前后向。autograd 保存物理 payload、scale 及恢复所需元数据，
让 saved-tensor hooks、CPU offload 和重算处理真实 tensor；不重新引入旧的融合 carrier 保存路径。

## GQA 投影与 TP

`MXFP8GQAAttention(GQAAttention)` 使用既有 `MXFP8Linear` 转换 `linear_qkv`、`o_proj`。
它继承基类 forward，保留 Norm、RoPE、cache、输出结构及 `attention_interface` 调用点。
投影仍通过子模块调用，既有可移除 hooks 继续有效。

QKV 行按 KV head 分组，每组为对应 Q heads、K、V；输出投影保持 `[H, Q_heads × Dv]`。
已加载和 meta 加载使用同一转换定义，保存时可还原源 Q/K/V 参数。
高精 bias 由既有 Linear 语义处理；TP 输出 bias 沿既有 deferred-bias 机制在归约后加一次。
低精投影仅声明 BF16；通信及 Attention 接收普通 BF16 张量。

TP 复用 planner 的 QKV `Shard(0)`、输出投影 `Shard(1)`，并调整本地 head 属性。
校验器遍历最终完整参数路径，要求每项权重恰有一份布局声明，并检查分片后的两个维度均为 32 的倍数。
全局维度对齐不能替代本地维度对齐。

GQA 为独立层调用声明 TP 复制值的反向规则：输入边界前向不变、反向 SUM；
输出边界前向 SUM、反向不变。该规则只作用于声明它的 Family，通用 collective 的 autograd 不变，
sequence-parallel 路径保留原有规则。

完整模型由 HSDP 管理梯度时，模型 builder 向 planner 传入 `source_gradient_reduction=true`，
关闭上述独立层专用规则，沿用现有输入/输出通信反向与 HSDP 源布局梯度归约及缩放。
这两个入口的 loss/梯度约定不同，不能叠加两套归约。
构建摘要的 `tp_gradient_contract` 记录 `replicated` 或 `source_layout`，
`tp_replicated_gradient_contract` 记录专用规则最终是否生效。
复制的 Norm 参数梯度继续由既有 HSDP 同步负责。
MXFP8 不支持 DTensor validate 模式；baseline GQA 的 TP validate 受 RoPE 混合张量分派缺口限制，
构建期明确报错。production 本地 Tensor 路径使用真实 planner 分片，不通过解除 DTensor 包装假装 validate。

GQA 没有登记 `fused_mxfp8` 变体。`use_fused_mxfp8=true` 会报错；
packed QKV、MXFP8 Linear 和普通 fused Attention 均不被解释为该开关的融合能力。

## 公共 GQA CP 协议

`attention_interface_cp_wrapper` 保留最终选定的 kernel。Q 保持本地；
K/V 使用现有可求导 all-gather，反向累加其他 rank 对本地 K/V 的贡献。

| 项目 | 默认协议 |
| --- | --- |
| Q/K/V | `[B, heads, local_S, D]`，普通 FP32/FP16/BF16 tensor |
| 分片 | 连续、等长、非空，按 CP rank 顺序拼接 |
| 位置 | RoPE 使用全局 position ids；因果 mask 加上 rank 的序列 offset |
| 输出 | 原接口返回 `[B, local_S, heads, Dv]`，保持本地 query 顺序 |
| mask | 全局坐标的布尔 allowed mask，或所选 kernel 支持的浮点 additive mask |
| dropout | 必须为零；其他设置需显式 RNG-aware adapter |
| cache、滑窗、packed | 默认协议拒绝，需专用模型 adapter |

二维 mask 支持 `[B, global_S]` padding 或 `[global_S, global_S]` 自定义矩阵；
维度有歧义时要求显式四维形式。四维 mask 的 batch/head 维须可广播，query 维为全局长度或 1。
默认 adapter 合并全局因果限制，将布尔 mask 规范为 additive mask 后交给原接口。
这不扩展原 kernel 对有限 additive bias 的支持范围。
默认 NPU kernel 在本地 Q 与全局 K 的长度不同时使用覆盖全局 K 的窗口，
由显式 mask 保持因果约束，避免额外的局部窗口截断合法的历史位置。

packed 长度参数在进入原 forward 和 attention interface 前均被拒绝，
包括 `actual_seq_len`、Q/KV 分别给定的长度、`cu_seq_lens_*`/`cu_seqlens_*`
及 `packed_seq_params` carrier；这些别名不会绕过连续等长分片的协议检查。

进入基类 forward 前，各 rank 交换 batch/sequence 形状，以便空分片或非等长分片一致报错，
避免其中一个 rank 跳过后续 K/V collective。该形状检查有同步开销，不代表完成了 CP 性能优化。
数据侧 labels、有效 token 数和 CE 归一化沿用生产 ParallelBatch/Trainer 路径。

## 其他 Attention Family

四类协议注册独立 adapter，当前默认组合限定 TP=PP=1；未登记其 FP8 变体。

| Family | 精确 baseline | CP 通信语义 |
| --- | --- | --- |
| MLA | `MLAAttention` | 保留压缩投影/RoPE，在展开后的 K/V 接口通信，梯度回传压缩 latent |
| DSA | `DeepseekV32DSAAttention` | 同步 indexer keys 和压缩 MLA keys，按本地 Q 的因果前缀构建稀疏索引 |
| Gated DeltaNet | Transformers `Qwen3_5GatedDeltaNet` | 因果卷积 halo、递归状态 P2P 及状态反向梯度 |
| Kimi Delta Attention | `KimiDeltaAttention` | 专用分块状态更新、卷积 halo、跨分片状态及反向梯度 |

MLA 当前不使用压缩 KV 的低通信量协议。DSA 需要 Omni 自定义稀疏算子，
其真实数值和 indexer 辅助 KL 的 CP 归一化尚未完成验收，不能据注册存在推断功能已验证。
另一个 `DSAAttention` 类型不因名称相近或继承关系自动加入这一 Family。

state 类复用既有完整层 CP executor，默认 eager backend、chunk size 64，
不会用 GQA all-gather 替代递归状态。旧模型 adapter 路径保留并委托公共实现。
KDA 的非因果 gate 差值在指数运算前屏蔽，避免 masked 位置溢出污染反向。

## 拓扑与范围

TP-extend-EP 由现有 mesh 构造器把非 PP dense domain 重排成 `(edp, ep)`。
TP=CP=EP=2 的示例使用 4 个 rank：dense 为 `(dp=1, cp=2, tp=2)`，expert 为 `(edp=2, ep=2)`。
EP 不是在 dense rank 数上再乘一次。实际 mesh、同步域和本地参数形状以构建输出为准。

本特性不声明 PP、HiFloat8、新 FP8 Attention kernel、低精通信、任意拓扑热迁移，
也不扩大为所有 FSDP/HSDP 组合已验收。具体设备验证范围以独立报告为准。

# Module Family：构建期规则选型设计

源码版本：`70c9574538b22f832f33f7c901cd6f3ba06ed3ac`，对应收缩后的 PR #1214。

## 1. 职责与接口

Module Family 将共享调用契约的实现组织为一个模块族。配置指定 baseline 和选择参数，构建阶段按完整规则确定具体实现，再调用已有 Module Replacement 完成源模块匹配、构造校验和实例安装。

| 组件 | 职责 |
| --- | --- |
| `ModuleFamily` | 声明 baseline、变体路径、选择参数和合法组合 |
| Family 注册表 | 保存经过校验的只读声明，按 baseline 对象身份查询 |
| `ReplacementTarget` | 保留原始入口、显式选择参数、选中 callable 和普通构造参数 |
| 配置 resolver | 为已注册 Family 解析选择参数；普通 target 沿用原解析流程 |
| 原 Module Replacement | 处理路径匹配、参数与接口契约、别名和实例安装 |

生产代码提供通用注册接口。具体实现与 Family 声明由模块所属组件提供，并在解析相应配置前完成注册。注册机制按精确对象身份查询，子类使用独立声明。当前提交的内置注册表为空，CPU 集成测试在隔离作用域中注册测试实现。

## 2. 声明与配置

### 2.1 数据结构

```python
@dataclass(frozen=True)
class SelectorSpec:
    value_type: type
    default: Any

@dataclass(frozen=True)
class VariantRule:
    when: Mapping[str, Any]
    variant: str

@dataclass(frozen=True)
class ModuleFamily:
    name: str
    baseline: Callable
    variants: Mapping[str, str]
    selector_specs: Mapping[str, SelectorSpec] = field(default_factory=dict)
    variant_rules: tuple[VariantRule, ...] = ()
```

`selector_specs` 声明参数的严格类型和 baseline 默认值。选择参数采用 `bool`、`int`、有限 `float` 或 `str`。分组、设备、构造上下文等复杂数据通过普通构造参数传递。

`variants` 将稳定的变体名映射到 Python 模块导入路径。`variant_rules` 将完整的具名选择参数组合映射到变体名。

### 2.2 接入方声明示例

下面的 `app.linear` 为接入方模块路径示例。相关类遵循原 `@module_replacement` 构造契约。

```python
register_module_family(ModuleFamily(
    name="linear",
    baseline=ReferenceLinear,
    variants={
        "matmul": "app.linear.MatmulLinear",
        "addmm": "app.linear.AddmmLinear",
    },
    selector_specs={
        "algorithm": SelectorSpec(str, "reference"),
        "fused": SelectorSpec(bool, False),
    },
    variant_rules=(
        VariantRule({"algorithm": "matmul", "fused": False}, "matmul"),
        VariantRule({"algorithm": "matmul", "fused": True}, "addmm"),
    ),
))
```

对应配置片段：

```yaml
plan_overrides:
  - match: "model.projection"
    module_type: torch.nn.Linear
    exact_type: true
    replace_module:
      _target_: app.linear.ReferenceLinear
      algorithm: matmul
      fused: true
```

`match` 和 `module_type` 选择模型中的源实例。`algorithm` 和 `fused` 选择该 Family 的实现类。规则声明位于 Python 注册代码中，训练配置只填写选择参数。

## 3. 白名单语义

允许的组合为 baseline 默认组合与全部显式规则的并集：

```text
允许集合 = {selector_specs 中的完整默认组合}
         ∪ {每条 VariantRule.when 的完整组合}
```

| 有效参数组合 | 行为 |
| --- | --- |
| `algorithm=reference, fused=false` | 选择 baseline |
| `algorithm=matmul, fused=false` | 选择 `MatmulLinear` |
| `algorithm=matmul, fused=true` | 选择 `AddmmLinear` |
| `algorithm=reference, fused=true` | 抛出配置错误 |
| 任意未登记的 algorithm 值 | 抛出配置错误 |

缺省字段先补齐默认值。类型检查采用精确 Python 类型，`1` 与 `true`、字符串 `"true"` 与布尔值 `true` 分别处理。完整条件按字段名和值比较，字典插入顺序不影响匹配。

默认值分支仅接受完整默认组合。其余配置必须匹配恰好一条规则；未命中时列出实际参数与允许组合并抛出错误。导入失败继续报错，规则优先级、部分匹配和静默回退均不参与选择。

### 3.1 注册校验

注册阶段检查声明名称、baseline 可调用性、参数名与标量类型、默认值、变体导入路径、条件字段完整性和变体引用。相同条件及重复声明 baseline 默认条件均拒绝。

选择参数不能占用 `_target_`、`target_path`、`module`、`module_fqn`、`context` 等配置或运行时注入字段。

通过校验后复制 variants、schema 和条件映射，以只读映射保存。选择参数使用不可变标量，调用方后续修改原字典不会改变注册结果。注册表保留 baseline 强引用，以对象身份区分具有自定义相等关系的 callable。

## 4. 构建流程

```text
解析 replace_module
    ↓
按 baseline 查询注册表
    ├─ 普通 target → 原 Target 解析与构造流程
    └─ 已注册 Family
          ↓
       提取选择参数、补齐默认值、检查类型
          ↓
       默认组合 / 完整规则匹配
          ↓
       延迟导入选中的变体
          ↓
       检查其余构造参数
          ↓
       ReplacementTarget 保存选择结果
          ↓
       原 factory 注入 module / module_fqn / context
          ↓
       原 replacement 校验并安装实例
```

Family 只在构建配置阶段消费选择参数。选型后的构造 kwargs 不包含 selector；其余键必须在最终 callable 的关键字签名中声明。运行时提供的源模块等参数允许在此阶段暂缺，通过 `bind_partial()` 检查已经配置的部分。

对未注册 Family 的 target，配置 resolver 和 factory 保留原行为，参数名称不会因为其他模块的 Family 声明而变成全局保留字。既有的命名 dtype scheme、条件过滤与 context 传递继续由原配置流程负责。

`trainer/config/parallelism.py` 对 Python 直接创建的普通 `Target` 查询 Family：仅已注册的 baseline 包装为 `ReplacementTarget`。传递过程中保留 Python 对象，避免通过序列化改变 callback 等参数。

原 `models/replacement.py` 继续负责具体模型对象的匹配、参数身份或权重转换校验、别名一致性与最终安装。Family 的选型层不改写这些校验和执行顺序。

## 5. 配置复制与覆盖

`ReplacementTarget` 同时保存以下信息：

| 字段 | 内容 |
| --- | --- |
| `requested_target` | 用户指定的 baseline callable |
| `selectors` | 用户显式填写的选择参数 |
| `callable` | 规则选择的具体实现 |
| `_kwargs` | 普通构造参数 |

```python
original = ...  # 已解析的 replacement Target
alternate = original.replace(algorithm="matmul")
fused = alternate.replace(fused=True)
baseline = fused.replace(algorithm="reference", fused=False)
```

每次 `.replace()` 合并参数后从原 baseline 重新选型，返回独立配置对象。多个相互依赖的字段可以一次性修改，最终组合仍通过白名单检查。

`to_dict()` 保存原 baseline 路径和用户显式填写的 selectors。缺省 selector 不补写，显式默认值保留。普通构造参数的默认值继续采用原配置解析器的行为。

Target 级路径覆盖将 selector 当作终止字段。`algorithm.extra` 等多余路径段报错。根配置列表索引语法保持原 resolver 能力边界。

## 6. 代码组织

| 文件 | 内容 |
| --- | --- |
| `hyper_parallel/components/modules/family.py` | 声明、注册校验、只读快照、白名单选择 |
| `hyper_parallel/trainer/config/replacement.py` | 配置意图、复制、序列化 |
| `hyper_parallel/trainer/config/resolver.py` | 已注册 Family 的配置解析与 selector 级覆盖 |
| `hyper_parallel/trainer/config/parallelism.py` | Python Target 到原 replacement factory 的接线 |
| `tests/ut/components/modules/test_module_family_selectors.py` | 白名单、类型、注册规则、快照和延迟导入 |
| `tests/ut/trainer/test_module_family_replacement.py` | 真实 CPU 模块替换、参数身份、别名、梯度与配置往返 |

CPU 集成测试在隔离注册表中使用 ReferenceLinear、MatmulLinear、AddmmLinear 三个测试实现，执行原模块替换引擎和真实 Torch 前反向。测试注册随用例清理；生产注册表保留给各模块组件显式接入。

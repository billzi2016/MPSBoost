# Native Multiclass Softmax 规格

## 目标

MPSBoost 的多分类最终默认实现必须是原生 multiclass softmax，而不是把多个二分类模型包装成最终方案。

OvR 可以保留为兼容层和 fallback，但不得在文档、测试或发布说明中声明为 native softmax。

## Public API 约束

用户接口保持 sklearn / XGBoost 风格：

- `fit(X, y, sample_weight=None)`
- `predict(X)`
- `predict_proba(X)`
- `decision_function(X)`
- `score(X, y, sample_weight=None)`
- `get_params(deep=True)`
- `set_params(**params)`
- `GridSearchCV` / `RandomizedSearchCV` / `cross_val_score`

底层从 OvR 切换到 native softmax 时，不应要求用户改训练代码。

## 策略参数

`multi_strategy` 取值：

- `auto`：默认策略。native softmax 可用时选择 native softmax；否则选择明确记录的 fallback。
- `softmax`：强制 native softmax。目标设备未实现 native softmax 时必须清晰提示，并报告实际兼容策略。
- `ovr`：显式使用 OvR compatibility layer。

当前阶段：

- CPU 多分类 `auto` 使用 native softmax。
- MPS native softmax 尚未完成时，不得把 CPU softmax 或 OvR 报告为 MPS softmax。

## 标签编码

Python 层负责把用户多分类标签编码为 `[0, num_class)` 的连续 class id。

模型必须保存或持有原始 `classes_`，预测时把 argmax class id 映射回用户标签。

## Base margin

CPU oracle 使用样本权重统计 class prior。

对 class `k`：

```text
base_score[k] = log(max(epsilon, weighted_count[k] / total_weight))
```

softmax 对统一平移不敏感，因此不要求 base score 归一到和为 0；但所有值必须有限。

## Softmax 概率

对每一行 raw margin `m`：

```text
p[k] = exp(m[k] - max(m)) / sum_j exp(m[j] - max(m))
```

每行概率必须满足：

- 全部有限；
- 全部非负；
- 行和约等于 1；
- `predict` 必须等于 `classes_[argmax(p)]`。

## Gradient / Hessian

每轮训练先基于当前所有 class margin 计算同一行的 softmax 概率。

对目标 class `k`：

```text
g[k] = p[k] - 1(label == k)
h[k] = p[k] * (1 - p[k])
```

这是 multiclass softmax 的 diagonal Hessian approximation；它不是独立 OvR binary logistic。

样本权重在 native 层统一乘到 gradient 和 Hessian：

```text
weighted_g = sample_weight * g
weighted_h = sample_weight * h
```

## Tree update 结构

CPU oracle 当前采用 round-major 结构：

```text
round 0: tree(class 0), tree(class 1), ..., tree(class K-1)
round 1: tree(class 0), tree(class 1), ..., tree(class K-1)
...
```

每棵树只更新一个 class margin，但该树的 gradient 来自同一轮全 class softmax 概率。

## 模型格式

native model header 使用显式 model kind 区分旧 regression/binary-logistic payload 与
multiclass softmax payload。

- `model kind = 0`：旧 regression/binary-logistic 格式，必须保持向后兼容读取。
- `model kind = 1`：native multiclass softmax 格式。

multiclass softmax payload 必须保存：

- `class_count`；
- `learning_rate`；
- 每个 class 的 `base_score`；
- Python public API 使用的 numeric `classes_` 映射；
- 训练时冻结的 quantization schema；
- round-major class tree/update 结构。

旧 regression/classifier loader 读取 `model kind = 1` 文件时必须明确失败，不得尝试把多分类
margin 文件解释为 regression 或 binary-logistic 模型。native softmax loader 读取旧
`model kind = 0` 文件时也必须明确失败，由 Python classifier 再进入 binary-logistic 兼容
加载路径。

## MPS 门槛

MPS native softmax 需要单独验收：

- 多 class margin buffer；
- softmax gradient/Hessian kernel；
- class-major 或 row-major histogram 输入 ABI；
- CPU oracle 对照；
- 真实数据集 parity；
- 明确性能报告。

在以上完成前，MPS 不得声明 native softmax 已完成。

当前阶段门槛：

- `multi_strategy="softmax", device="mps"` 必须明确提示并使用 OvR compatibility strategy，
  并在训练摘要中显示实际策略；
- `multi_strategy="auto", device="mps"` 可以使用 OvR compatibility fallback，但训练摘要必须
  显示 `strategy = "one_vs_rest"`；
- CPU `auto` 与 CPU `softmax` 必须使用 native softmax；
- MPS parity 测试在 native softmax kernels 完成前只能比较 CPU OvR 与 MPS OvR，不得把
  CPU native softmax 当作 MPS native softmax 对照。

## 验收

完成 native softmax 的最低验收：

- objective helper 直接测试 softmax 概率、gradient、Hessian；
- CPU native softmax estimator 不产生 `estimators_` OvR 子模型；
- `multi_strategy="softmax", device="mps"` 明确提示并使用 OvR compatibility strategy，
  并报告实际 strategy；
- 保存/加载后 class mapping、raw margin、probability 和 predict 一致；
- `predict_proba` 行归一；
- `GridSearchCV` 可调参；
- Iris / Digits / Covertype subset 覆盖默认 CPU native softmax 路径；
- Covertype subset 的 MPS 当前行为覆盖 OvR compatibility fallback；
- OvR 仍可通过 `multi_strategy="ovr"` 显式启用。

# 模块设计：训练核心

## 1. 职责

训练核心定义目标函数、split gain、叶子权重、树结构和 boosting 状态机。它只依赖抽象 `ComputeBackend`，不依赖 Python、Metal 对象、缓存路径或文件系统。

## 2. 唯一数学语义

平方误差：

```text
g_i = prediction_i - label_i
h_i = 1
```

节点分数、叶值和切分增益：

```text
score(G, H) = G² / (H + λ)
weight(G, H) = -G / (H + λ)
gain = 0.5 × (score_left + score_right - score_parent) - γ
```

所有实现调用共享的领域函数或严格对照它的 kernel 契约，禁止在 Python、CPU 和 GPU 各维护一份不同公式。

## 3. 树模型

扁平节点必须包含：特征索引、threshold bin、左右子节点索引、叶值、gain 和 flags。叶节点通过 flag 区分，不使用特殊浮点值。所有索引在创建和加载时验证。

稳定 tie-break 顺序：

1. gain 更大；
2. feature index 更小；
3. threshold bin 更小；
4. 其他条件完全相同则保持首次候选。

浮点接近时仍按冻结比较规则执行，不能依赖线程完成顺序。

冻结比较规则为 FP64 `gain` 严格 `>`；只有位级计算结果相等才进入 feature/bin
tie-break，不使用随数据尺度改变含义的 epsilon。候选 gain 必须严格大于 0，零或负值
保持叶节点。分箱值 `bin <= threshold_bin` 进入左子树，其余进入右子树。

训练参数必须满足：`min_samples_leaf >= 1`、`min_child_weight >= 0`、
`reg_lambda >= 0`、`gamma >= 0`，所有浮点参数必须有限。左右子节点必须同时满足
最小样本、最小 Hessian 和严格正 Hessian；违反约束的候选不得进入 gain 比较。

## 4. 后端接口

```text
ComputeBackend
├── compute_gradients(...)
├── build_histograms(...)
├── update_predictions(...)
├── synchronize()
└── diagnostics()
```

后端接口传递稳定 POD 视图和预分配输出。训练核心负责何时调用，后端负责如何计算。后续 split/partition 上 GPU 时扩展细粒度能力接口，不让核心接触 command buffer。

## 5. 状态机

```text
Created → Validated → Quantized → BackendReady
        → Iterating → ModelFinalized → Completed
                         ↘ Failed
```

- 每次状态转移验证前置条件；
- `Failed` 不得导出模型；
- estimator 只有 `Completed` 后才原子接收模型；
- 训练会话拥有全部临时资源，异常路径统一释放。

## 6. 树生长

0.2.0 按层生长深度受限树：

1. 获取当前层活跃节点；
2. 构建节点直方图；
3. 按稳定规则选 split；
4. 检查最小样本和 Hessian；
5. 分区样本；
6. 创建下一层或冻结叶节点；
7. 树完成后更新训练预测。

按层策略便于批处理多个节点，是唯一默认策略；不得同时保留另一套无规格的深度优先生产逻辑。

## 7. CPU oracle

CPU backend 以清晰和确定性为首要目标，用 FP64 累计形成 oracle。它不是隐藏 fallback。用户选择 `mps` 后，后端不可用必须失败；训练核心可以按已规格化策略让 CPU 承担控制流，但不能把 GPU 热路径整体替换掉仍报告 `mps` 成功。

CPU histogram 按传入 row index 顺序累计每个 bin 的 `count/G/H`，树节点按 breadth-first
顺序写入扁平数组。树生长控制流只存在于 core；CPU backend 只构建 histogram，不能
复制 split 选择、分区或节点组装逻辑。

## 8. 验收

- 手算小树逐节点一致；
- 多轮 boosting 预测更新正确；
- 同参数重复训练满足确定性约束；
- 后端异常不产生半模型；
- 核心测试不需要 Python 或真实 GPU，GPU 集成另行真实验证。

# 模块设计：Python API

## 1. 职责

提供简单、稳定的 estimator 风格入口，完成参数验证、输入适配、异常转换和结果呈现。该层不得实现分箱、树生长或 GPU 热路径。

## 2. 公共入口

```python
import mpsboost as mb

model = mb.GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    max_bins=256,
    min_child_weight=1.0,
    min_samples_leaf=20,
    reg_lambda=1.0,
    random_state=None,
    device="mps",
    verbosity=1,
)
model.fit(X, y)
prediction = model.predict(X_test)
```

0.2.x 公共符号仅包含已完成能力：`GradientBoostingRegressor`、`MPSBoostRegressor`、`is_available`、`system_info`、`__version__` 以及缓存诊断/管理函数。未实现的分类和底层训练 API 不得放入正式公共入口。

## 3. 构造契约

- 构造函数只保存参数，无设备初始化、文件写入或数据分配。
- 显式参数优于 `**kwargs`；未知参数由 Python 自然报错。
- `get_params()` 返回全部构造参数；`set_params()` 只接受已知名称并返回 `self`。
- 参数语义只在一个验证器中定义，不在 estimator、binding 和 C++ 各复制一份。

## 4. fit 契约

### 输入

- `X`：二维数值稠密数组；
- `y`：长度等于行数的一维数值数组；
- 0.2.0 不接受 NaN、Inf、稀疏矩阵、类别 dtype 或多输出。

### 行为

1. Python 层做轻量类型和结构检查。
2. native 边界做完整尺寸、溢出和内存布局检查。
3. 创建训练会话并在长计算期间释放 GIL。
4. 成功后原子替换 estimator 模型状态；失败保留未拟合状态。
5. 返回 `self`。

### 异常

- 输入与参数错误：`ValueError` 或 `TypeError`；
- MPS 不可用：`MPSBackendUnavailable`；
- 内存预算不足：`MPSBoostMemoryError`；
- 设备 command 失败：`MPSExecutionError`；
- 未拟合预测：`NotFittedError`。

异常由统一转换层创建，消息包含失败阶段和解决建议，不暴露实现栈中的无意义错误码。

## 5. predict 契约

- 输入特征数必须与训练一致；
- 返回一维 `float32` 或文档冻结的 dtype；
- CPU 与 MPS 推理读取同一扁平模型；
- 预测不得修改模型或输入；
- 小批量策略必须唯一，不能存在不透明的多套预测逻辑。

## 6. 拟合后状态

- `n_features_in_`
- `device_`
- `n_estimators_`
- `model_`（私有 native handle，不作为可序列化 Python 状态）
- `training_summary_`（非敏感耗时和内存摘要）

## 7. 线程与所有权

- 同一 estimator 的并发 `fit` 不支持并明确拒绝；
- 多个 estimator 可共享只读 pipeline cache，但不共享可变训练状态；
- native model 用明确所有权对象管理，Python 析构不得在未同步 GPU 工作时释放资源。

## 8. 注释与测试

- 每个公共方法必须有中文契约 docstring；
- 测试 `get/set_params`、重复 fit、失败原子性、未拟合、输入生命周期和 GIL 行为；
- 不得用假 native handle 或 mock GPU 验收公共训练流程。

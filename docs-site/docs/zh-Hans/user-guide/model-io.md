# 模型保存与加载

MPSBoost 使用 versioned native model format 保存模型。

## 支持范围

当前支持：

- numeric regression model
- binary classifier model
- native CPU multiclass softmax model
- 高级回归 objective metadata

## 设计约束

保存文件必须包含：

- model structure
- version metadata
- objective metadata
- feature/bin schema
- class mapping，适用于 classifier

保存文件不得包含：

- training data
- credentials
- telemetry
- device identifiers

加载时会进行兼容性校验，防止用错误 estimator 或错误 objective 读取模型。

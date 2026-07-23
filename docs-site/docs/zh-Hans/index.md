# MPSBoost 中文文档

MPSBoost 是面向 Apple Silicon 的树模型学习库。当前公开版本为 `1.0.0`，包含 native CPU/MPS 后端、sklearn 风格 estimator、多分类 native CPU softmax、随机森林、ExtraTrees、DecisionTree、CatBoost-like numeric estimator、高级回归目标、解释工具、异常检测和排序学习入口、可选 SHAP/portable-backend 诊断、真实世界验收报告、性能报告，以及客户侧 fallback 加固。

本中文文档站点以根目录 README、规格和发布审计为单一事实源。已有项目文件通过 symlink 接入，避免复制后产生内容漂移。

## 快速入口

- [安装与环境](getting-started/install.md)
- [最小示例](getting-started/quick-start.md)
- [后端策略](user-guide/backend-policy.md)
- [Estimator API](user-guide/estimators.md)
- [模型族](model-families/gradient-boosting.md)
- [发布总览](release/index.md)
- [1.0.0 发布](release/1.0.0.md)
- [1.0.0 发布审计](project/RELEASE_AUDIT_1.0.0.md)
- [任务清单](specs/tasks.md)

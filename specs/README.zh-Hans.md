# MPSBoost SDD 规格索引

本目录是项目唯一规格来源。`constitution.md` 是共享最高工程宪法，`project-tree.md`
是共享目录规范，`tasks.md` 是共享总任务清单，`v1-mps-histogram-engine/` 保存
0.2.0 MPS histogram engine 基础规格，`v2-arboretum-implementation/` 保存后续全树模型
路线，`v3-real-world-tests/` 保存进入 1.x 前的真实世界数据集验收门。实现前必须先
阅读宪法、对应版本规格和任务清单；规格之间若有冲突，以当前用户指令和宪法为最高
约束，并先修正规格后编码。

## 共享规格

| 文件 | 作用 |
| --- | --- |
| [constitution.md](constitution.md) | 不可绕过的共享工程宪法与质量门 |
| [project-tree.md](project-tree.md) | 共享目录结构和文件职责 |
| [tasks.md](tasks.md) | 有依赖、有验收条件的实施清单 |
| [v2-arboretum-implementation/prd.md](v2-arboretum-implementation/prd.md) | v2 全树模型族和工业表格能力路线 |
| [v3-real-world-tests/prd.md](v3-real-world-tests/prd.md) | v3 真实世界数据集测试与 1.x 发布门 |

## V1 MPS Histogram Engine 核心规格

| 文件 | 作用 |
| --- | --- |
| [v1-mps-histogram-engine/prd.md](v1-mps-histogram-engine/prd.md) | 用户问题、产品范围、功能与非功能需求 |

## V1 MPS Histogram Engine 模块设计

| 文件 | 模块 |
| --- | --- |
| [v1-mps-histogram-engine/modules/01-python-api.md](v1-mps-histogram-engine/modules/01-python-api.md) | estimator 风格 Python API |
| [v1-mps-histogram-engine/modules/02-data-quantization.md](v1-mps-histogram-engine/modules/02-data-quantization.md) | 输入、分箱和数据所有权 |
| [v1-mps-histogram-engine/modules/03-training-core.md](v1-mps-histogram-engine/modules/03-training-core.md) | 目标函数、树生长和训练状态机 |
| [v1-mps-histogram-engine/modules/04-mps-backend.md](v1-mps-histogram-engine/modules/04-mps-backend.md) | MPS/Metal 运行时与 kernel |
| [v1-mps-histogram-engine/modules/05-cache.md](v1-mps-histogram-engine/modules/05-cache.md) | 三层缓存与失效策略 |
| [v1-mps-histogram-engine/modules/06-model-io.md](v1-mps-histogram-engine/modules/06-model-io.md) | 模型格式、加载和兼容性 |
| [v1-mps-histogram-engine/modules/07-quality.md](v1-mps-histogram-engine/modules/07-quality.md) | 测试、正确性、基准和安全 |
| [v1-mps-histogram-engine/modules/08-packaging-release.md](v1-mps-histogram-engine/modules/08-packaging-release.md) | wheel、CI、Git 与 PyPI 发布 |

## 执行规则

1. 不得跳过规格直接实现。
2. 不得存在两套同类逻辑或临时 mock。
3. 每个任务只有在代码、注释、测试和验收全部完成后才能打勾。
4. 模块设计若需改变，先更新关联规格并说明影响范围。

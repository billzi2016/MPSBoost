# 项目目录规范

以下是目标唯一目录结构。未经规格变更，不新增平行实现目录。

```text
MPSBoost/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── release.yml
├── AGENTS.md                    # 本地 Agent 规则；Git 忽略
├── specs/                       # 本地 SDD 规格；Git 忽略
│   ├── README.md                # 共享规格索引
│   ├── constitution.md          # 共享工程宪法
│   ├── project-tree.md          # 共享目录结构
│   ├── tasks.md                 # 共享总任务清单
│   ├── v2-arboretum-implementation/ # v2 全树模型路线
│   │   └── prd.md
│   ├── v3-real-world-tests/     # v3 真实世界数据集测试与 1.x 发布门
│   │   └── prd.md
│   └── v1-mps-histogram-engine/
│       ├── prd.md
│       └── modules/
│           ├── 01-python-api.md
│           ├── 02-data-quantization.md
│           ├── 03-training-core.md
│           ├── 04-mps-backend.md
│           ├── 05-cache.md
│           ├── 06-model-io.md
│           ├── 07-quality.md
│           └── 08-packaging-release.md
├── include/mpsboost/
│   ├── backend.hpp              # 计算后端抽象
│   ├── binned_dataset.hpp       # 量化数据只读视图
│   ├── objective.hpp            # 目标函数接口
│   ├── tree.hpp                 # 稳定领域模型
│   ├── trainer.hpp              # 训练状态机
│   └── version.hpp
├── src/
│   ├── core/
│   │   ├── binned_dataset.cpp
│   │   ├── objective.cpp
│   │   ├── tree.cpp
│   │   └── trainer.cpp
│   ├── cpu/
│   │   └── reference_backend.cpp
│   ├── mps/
│   │   ├── mps_backend.mm
│   │   ├── metal_context.mm
│   │   ├── buffer_pool.mm
│   │   └── kernels/
│   │       ├── gradients.metal
│   │       ├── histogram.metal
│   │       ├── split_scan.metal
│   │       ├── partition.metal
│   │       └── prediction.metal
│   ├── io/
│   │   └── model_format.cpp
│   ├── python/
│   │   └── bindings.cpp
│   └── mpsboost/
│       ├── __init__.py
│       ├── estimator.py
│       ├── matrix.py
│       ├── booster.py
│       ├── cache.py
│       └── diagnostics.py
├── tests/
│   ├── unit/
│   ├── metal/
│   ├── integration/
│   ├── packaging/
│   └── benchmarks/
├── benchmarks/
│   ├── datasets.py
│   ├── runner.py
│   ├── report.py
│   └── results/                 # 已验证原始 benchmark 与可读摘要；不进入 wheel
├── CMakeLists.txt
├── pyproject.toml
├── README.md
├── CHANGELOG.md
├── LICENSE
└── .gitignore
```

## 目录职责约束

- `include/mpsboost`：稳定 C++ 契约，不包含 Objective-C 类型。
- `src/core`：设备无关领域逻辑，不访问 Python、Metal 或文件系统。
- `src/cpu`：唯一 CPU oracle；只为正确性和显式 CPU 模式服务。
- `src/mps`：设备资源与 kernel，不能重新定义训练数学语义。
- `src/io`：模型格式，不依赖训练会话。
- `src/python`：薄绑定，不复制 Python 层参数逻辑。
- `src/mpsboost`：用户体验、参数验证和异常转换，不实现热路径。
- `tests`：按真实边界分层；共享测试数据生成器，不复制期望算法。
- `benchmarks`：与测试分开，不能参与包运行时。

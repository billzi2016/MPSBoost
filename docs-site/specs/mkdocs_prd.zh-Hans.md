# MPSBoost MkDocs 中文文档站点 PRD

## 1. 文档目的

本 PRD 定义 MPSBoost 文档站点的结构、内容来源、MkDocs 配置和维护约束。目标是建设一个双语项目文档站点，准确展示当前 `1.0.0` 能力，并与根目录 README、规格、任务清单和版本化发布审计保持一致。

本阶段不做中英双语站点。所有 PRD、导航和初始页面均使用 `zh-Hans` 后缀或目录语义，避免后续语言扩展时命名混乱。

## 2. 建设目标

- 在 `docs-site/` 下建立独立 MkDocs 工程。
- 默认语言为简体中文，目录使用 `zh-Hans/`。
- PRD 文件命名使用 `.zh-Hans.md` 后缀。
- 已有仓库文档必须通过 symlink 接入，不复制。
- 导航覆盖 MPSBoost 当前公开能力、快速开始、后端策略、发布审计和规格文档。
- 站点内容必须符合当前 `1.0.0` 项目状态，并保持 append-only release history。
- 为后续英文或其他语言扩展保留结构，但本阶段不生成英文占位内容。

## 3. 技术选型

- 文档框架：MkDocs
- 主题：Material for MkDocs
- 默认语言：`zh-Hans`
- Markdown 扩展：`admonition`、`tables`、`toc`、`pymdownx.highlight`、`pymdownx.superfences`
- 部署：GitHub Pages

文档依赖写入 `docs-site/requirements.txt`，不加入根项目运行时依赖。

## 4. 目录结构

应建立以下结构：

```text
docs-site/
├── mkdocs.yml
├── requirements.txt
├── assets/
├── overrides/
├── specs/
│   ├── github_action_prd.zh-Hans.md
│   └── mkdocs_prd.zh-Hans.md
└── docs/
    └── zh-Hans/
        ├── index.md
        ├── getting-started/
        ├── user-guide/
        ├── project/
        └── specs/
```

其中 `docs-site/specs/` 放站点建设 PRD；`docs-site/docs/zh-Hans/specs/` 通过 symlink 接入根目录 `specs/` 中已有项目规格。

## 5. 内容来源规则

### 5.1 必须 symlink 的已有文件

以下已有文件接入文档站点时必须使用 symlink：

- `README.md`
- `docs/CHANGELOG.md`
- `docs/RELEASE_AUDIT_*.md`
- `ai-skills/mps_boost_skill.md`
- `specs/tasks.md`
- `specs/constitution.md`
- `specs/project-tree.md`
- `specs/AGENTS.md`
- `specs/v1-mps-histogram-engine/prd.md`
- `specs/v1-mps-histogram-engine/modules/*.md`
- `specs/v2-arboretum-implementation/prd.md`
- `specs/v2-arboretum-implementation/native-multiclass-softmax.md`
- `specs/v3-real-world-tests/prd.md`

不得把这些文件复制到 `docs-site/docs/`。

### 5.2 可以新建的文件

只允许新建站点工程自身文件：

- `docs-site/mkdocs.yml`
- `docs-site/requirements.txt`
- `docs-site/docs/zh-Hans/index.md`
- `docs-site/docs/zh-Hans/getting-started/*.md`
- `docs-site/docs/zh-Hans/user-guide/*.md`
- `docs-site/specs/*.zh-Hans.md`

新建页面应简洁，优先作为导航入口，不重复已有长文档内容。

## 6. 导航要求

导航必须以当前项目能力为准，至少包含：

- 首页
- 快速开始
- 安装与环境诊断
- 后端选择策略
- Estimator API
- Release overview 和带版本发布审计
- Changelog
- 项目规格
- 文档站点 PRD

导航中引用已有文件时，应引用 symlink 路径。

## 7. 当前项目状态要求

文档必须反映 `1.0.0` 状态：

- `mpsboost==1.0.0` 是当前发布目标。
- Apple Silicon wheel 支持 `cp313` / `macosx_13_0_arm64`。
- MPSBoost native CPU/MPS 后端继续是核心实现。
- CPU backend 是 correctness oracle，不因 S22 portable backend 规划被替代。
- 已有模型族包括 GBDT 回归/分类、多分类 native CPU softmax、DecisionTree、RandomForest、ExtraTrees、CatBoost-like numeric、IsolationForest、LearningToRankRegressor。
- isolation/ranking 是 CPU-suitable workflow；请求 `device="mps"` 时应提示 CPU 更适合并继续运行。
- 环境缺失时给出复制即用安装命令和 `MPSBOOST_SKIP_ENV_CHECK=1` 跳过方式。
- S22 portable-backend 诊断和策略作为显式可选 adapter guidance 交付，不作为隐藏 native replacement。

## 8. MkDocs 配置要求

`docs-site/mkdocs.yml` 必须：

- 设置 `site_name: MPSBoost`
- 设置中文站点语言。
- 使用 Material 主题。
- 明确 `docs_dir: docs`
- 明确 `site_dir: site`
- 配置导航。
- 启用严格构建所需的基础 Markdown 扩展。
- 不引用不存在的页面。

## 9. 本地验证

推荐本地命令：

```bash
python -m pip install -r docs-site/requirements.txt
mkdocs build --config-file docs-site/mkdocs.yml --strict
```

如果执行环境不允许安装依赖，应至少检查 symlink、配置文件和导航路径一致性。

## 10. 验收标准

完成条件：

1. 两个站点 PRD 均为 `.zh-Hans.md` 文件。
2. `docs-site/` 目录骨架完整。
3. MkDocs 配置存在且符合当前 MPSBoost 项目。
4. 已有项目文档通过 symlink 接入。
5. 没有复制 README、CHANGELOG、RELEASE_AUDIT、根目录 `specs/` 等源文件。
6. 中文首页和基础入口页存在。
7. 本地 `mkdocs build --strict` 可以通过，或明确记录因依赖缺失无法运行。
8. GitHub Pages workflow 可使用该站点配置构建。

## 11. 执行指令

实现者应直接完成：

- 重命名原 `github_action_prd.md` 和 `mkdocs_prd.md` 为 `.zh-Hans.md` 语义文件。
- 建立 `docs-site` 目录骨架。
- 新增 MkDocs 配置和依赖文件。
- 将已有文档以 symlink 接入。
- 新增最少必要中文入口页。
- 不处理 S20 程序文件中文清理。

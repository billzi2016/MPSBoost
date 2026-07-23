# MPSBoost 文档站点 GitHub Actions PRD

## 1. 文档目的

本 PRD 定义 MPSBoost 文档站点的 GitHub Actions 构建与部署要求。目标是让文档站点可以稳定发布到 GitHub Pages，同时不干扰 MPSBoost 主项目的 native wheel、PyPI 发布、真实 MPS 测试和自托管 runner 流程。

本文件只约束 `docs-site/` 文档工程，不替代根目录 `.github/workflows/ci.yml` 的包构建和测试职责。

## 2. 建设目标

- 为 `docs-site/` 增加独立 GitHub Pages 部署工作流。
- 只在文档站点、公开文档入口或工作流自身变化时触发。
- 使用 MkDocs 构建静态站点。
- 使用 GitHub 官方 Pages Actions 上传并部署产物。
- 采用最小权限，避免获得 PyPI、包发布或 self-hosted GPU runner 权限。
- 支持手动触发，用于补发部署和排查 Pages 问题。

## 3. 适用范围

适用于：

- `docs-site/` 下 MkDocs 站点构建。
- GitHub Pages 静态部署。
- README、`docs/`、`ai-skills/`、根目录 `specs/` 等文档源文件变化后的站点更新。
- 由 symlink 接入的既有文档源文件。

不适用于：

- PyPI 发布。
- wheel 构建和上传。
- self-hosted MPS/GPU 测试。
- 大型真实数据集下载。
- 任何需要写入凭据、修改 tag 或上传 release artifact 的流程。

## 4. 工作流文件

应新增独立工作流：

- 路径：`.github/workflows/docs.yml`
- 名称：`Docs`
- 部署目标：GitHub Pages

不得把文档部署逻辑塞进现有 `ci.yml`。现有 CI 继续负责 Python/native 构建和测试；docs workflow 只负责站点。

## 5. 触发策略

必须支持：

- `workflow_dispatch`
- `push` 到 `main`

`push` 必须使用 `paths` 限定触发范围，至少包含：

- `.github/workflows/docs.yml`
- `docs-site/**`
- `README.md`
- `docs/CHANGELOG.md`
- `docs/RELEASE_AUDIT_*.md`
- `ai-skills/mps_boost_skill.md`
- `specs/**`

不应因为普通源码、测试或 benchmark 改动触发文档部署，除非这些改动同时修改了上述文档入口。

## 6. 权限要求

工作流权限必须最小化：

- `contents: read`
- `pages: write`
- `id-token: write`

不得授予 PyPI token、packages write、actions write、contents write 或 self-hosted runner 专属能力。

## 7. 并发控制

必须配置 Pages 部署并发组，避免同一分支重复部署互相覆盖：

- group 可使用 `pages`
- `cancel-in-progress` 可为 `false`，保证正在发布的 Pages job 不被中途取消

## 8. 构建环境

- 使用 `ubuntu-latest`。
- 使用稳定 Python 版本，建议 `3.12`。
- 依赖安装以 `docs-site/requirements.txt` 为准。
- 构建命令在 `docs-site/` 目录执行。

推荐命令：

```bash
python -m pip install --upgrade pip
python -m pip install -r docs-site/requirements.txt
mkdocs build --config-file docs-site/mkdocs.yml --strict --site-dir site
```

## 9. 部署步骤

工作流应使用 GitHub 官方 Pages Actions：

- `actions/checkout`
- `actions/setup-python`
- `actions/configure-pages`
- `actions/upload-pages-artifact`
- `actions/deploy-pages`

部署 artifact 应来自 MkDocs 构建输出目录，不提交构建产物到仓库。

## 10. Symlink 约束

MPSBoost 已有公开文档和规格文件必须通过 symlink 接入 `docs-site/docs/zh-Hans/`，不能复制内容。

原因：

- README、`docs/`、`ai-skills/`、`specs/` 是单一事实源。
- 复制会导致 PyPI README、GitHub README、文档站点内容漂移。
- 后续 agent 修改源文件后，站点应自动使用最新内容。

GitHub Actions checkout 必须保留 symlink。不得在 workflow 中把 symlink 展开为复制文件。

## 11. 验收标准

完成条件：

1. `.github/workflows/docs.yml` 存在。
2. workflow 只覆盖文档站点构建和 GitHub Pages 部署。
3. workflow 支持 `push` path filter 和 `workflow_dispatch`。
4. workflow 使用最小 Pages 权限。
5. `mkdocs build --strict` 可在本地或 CI 中执行。
6. `docs-site/docs/zh-Hans/` 中接入既有文件时使用 symlink。
7. workflow 不访问 PyPI token，不构建 wheel，不运行 GPU 测试。
8. 失败时能直接暴露 MkDocs 配置、链接或 Markdown 错误。

## 12. 执行指令

实现者应基于本 PRD：

- 新增 `.github/workflows/docs.yml`。
- 保持根 CI 不变，除非确有必要且单独说明。
- 使用 `docs-site/requirements.txt` 安装文档依赖。
- 使用 `docs-site/mkdocs.yml` 构建。
- 部署到 GitHub Pages。
- 不复制已有文档源文件；全部通过 symlink 引入。

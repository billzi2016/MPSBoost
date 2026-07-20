# 文档站维护

文档站位于 `docs-site/`。

## 本地构建

```bash
python -m pip install -r docs-site/requirements.txt
mkdocs build --config-file docs-site/mkdocs.yml --strict
```

## 文件原则

- 已有项目 Markdown 必须 symlink 接入。
- 不复制 README、CHANGELOG、RELEASE_AUDIT、mps_boost_skill、根目录 `specs/`、benchmark 报告或测试 README。
- 新页面只用于站点入口、导航说明和用户指南。
- `docs-site/site/` 是构建产物，已加入 `.gitignore`。

## GitHub Pages

`.github/workflows/docs.yml` 只负责文档站构建和部署，不运行 package CI、不发布 PyPI、不占用 MPS GPU 测试资源。

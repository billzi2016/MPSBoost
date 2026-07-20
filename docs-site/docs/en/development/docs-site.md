# Documentation Site Maintenance

The documentation site lives in `docs-site/`.

## Local Build

```bash
python -m pip install -r docs-site/requirements.txt
mkdocs build --config-file docs-site/mkdocs.yml --strict
```

## File Principles

- Existing project Markdown is connected with symlinks.
- Do not copy README, CHANGELOG, RELEASE_AUDIT, `mps_boost_skill`, root `specs/`,
  benchmark reports, or test README files.
- New pages provide site entry points, navigation, and user guides.
- `docs-site/site/` is ignored build output.

The Docs workflow builds and deploys only the site; it does not run package CI,
publish PyPI, or consume MPS GPU test resources.

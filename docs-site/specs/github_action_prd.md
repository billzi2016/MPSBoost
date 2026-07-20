# MPSBoost Documentation Site GitHub Actions PRD

## 1. Documentation Purpose

This PRD defines GitHub Actions build and deployment requirements for the MPSBoost
documentation site. The goal is stable GitHub Pages publication without interfering
with the main project's native wheels, PyPI releases, real MPS tests, or self-hosted
runner workflows.

This file constrains only the `docs-site/` documentation project. It does not
replace the root `.github/workflows/ci.yml` package-build and test responsibilities.

## 2. Construction Goals

- Add an independent GitHub Pages deployment workflow for `docs-site/`.
- Trigger only when the documentation site, public documentation entry points, or the
  workflow itself changes.
- Build a static site with MkDocs.
- Upload and deploy artifacts with official GitHub Pages Actions.
- Use least privilege; do not receive PyPI, package-release, or self-hosted GPU-runner permission.
- Support manual triggering for redeployment and Pages troubleshooting.

## 3. Scope

Applies to:

- MkDocs site builds under `docs-site/`.
- GitHub Pages static deployment.
- Site updates after changes to README, CHANGELOG, RELEASE_AUDIT, root `specs/`,
  and other documentation sources.
- Existing documentation sources included through symlinks.

Does not apply to:

- PyPI publication.
- Wheel building and upload.
- Self-hosted MPS/GPU tests.
- Large real-dataset downloads.
- Any process that writes credentials, changes tags, or uploads release artifacts.

## 4. Workflow File

Add an independent workflow:

- Path: `.github/workflows/docs.yml`
- Name: `Docs`
- Deployment target: GitHub Pages

Do not put document-deployment logic in existing `ci.yml`. Existing CI continues to
build and test Python/native code; the docs workflow owns the site only.

## 5. Trigger Policy

Support:

- `workflow_dispatch`
- push to `main`

Push must use a `paths` filter containing at least:

- `.github/workflows/docs.yml`
- `docs-site/**`
- `README.md`
- `CHANGELOG.md`
- `RELEASE_AUDIT.md`
- `mps_boost_skill.md`
- `specs/**`

Ordinary source, test, or benchmark changes must not deploy documents unless they
also modify one of these documentation entries.

## 6. Permission Requirements

Workflow permissions are least privilege:

- `contents: read`
- `pages: write`
- `id-token: write`

Do not grant PyPI tokens, packages write, actions write, contents write, or
self-hosted-runner-specific capabilities.

## 7. Concurrency Control

Configure a Pages-deployment concurrency group so repeated deployments from one branch
do not overwrite one another:

- the group may use `pages`;
- `cancel-in-progress` may be `false` so an ongoing Pages job is not cancelled.

## 8. Build Environment

- Use `ubuntu-latest`.
- Use a stable Python version, preferably `3.12`.
- Install dependencies from `docs-site/requirements.txt`.
- Run build commands in `docs-site/`.

Recommended commands:

```bash
python -m pip install --upgrade pip
python -m pip install -r docs-site/requirements.txt
mkdocs build --config-file docs-site/mkdocs.yml --strict --site-dir site
```

## 9. Deployment Steps

The workflow uses official GitHub Pages Actions:

- `actions/checkout`
- `actions/setup-python`
- `actions/configure-pages`
- `actions/upload-pages-artifact`
- `actions/deploy-pages`

Deployment artifacts originate from the MkDocs build-output directory; do not commit
build artifacts to the repository.

## 10. Symlink Constraints

Existing MPSBoost public documentation and specifications must enter
`docs-site/docs/zh-Hans/` through symlinks, not copied content.

Reasons:

- README, CHANGELOG, RELEASE_AUDIT, and `specs/` are single sources of truth.
- Copies cause drift between PyPI README, GitHub README, and documentation-site content.
- When future agents modify source files, the site automatically uses current content.

GitHub Actions checkout must retain symlinks. Do not expand them into copied files in
the workflow.

## 11. Acceptance Criteria

Completion requires:

1. `.github/workflows/docs.yml` exists.
2. The workflow covers only documentation-site build and GitHub Pages deployment.
3. The workflow supports push path filters and `workflow_dispatch`.
4. The workflow uses least-privilege Pages permissions.
5. `mkdocs build --strict` runs locally or in CI.
6. Existing files enter `docs-site/docs/zh-Hans/` through symlinks.
7. The workflow neither accesses PyPI tokens nor builds wheels nor runs GPU tests.
8. Failures directly expose MkDocs configuration, link, or Markdown errors.

## 12. Execution Instructions

Implementers must use this PRD to:

- add `.github/workflows/docs.yml`;
- leave root CI unchanged unless truly necessary and separately explained;
- install documentation dependencies from `docs-site/requirements.txt`;
- build with `docs-site/mkdocs.yml`;
- deploy to GitHub Pages;
- do not copy existing documentation sources; include all of them through symlinks.

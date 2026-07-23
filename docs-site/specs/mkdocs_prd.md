# MPSBoost MkDocs Chinese Documentation Site PRD

## 1. Documentation Purpose

This PRD defines the MPSBoost documentation site's structure, content sources,
MkDocs configuration, and maintenance constraints. The goal is a Simplified Chinese
project documentation site that accurately presents current `0.5.0` capabilities
and remains consistent with the root README, specifications, task list, and release audit.

This stage does not build a bilingual Chinese/English site. Every PRD, navigation
entry, and initial page uses a `zh-Hans` suffix or directory semantic to prevent
naming confusion during later language expansion.

## 2. Construction Goals

- Establish an independent MkDocs project under `docs-site/`.
- Use Simplified Chinese by default and the `zh-Hans/` directory.
- Use the `.zh-Hans.md` suffix for PRD files.
- Bring existing repository documentation in through symlinks, never copies.
- Cover current public MPSBoost capability, quick start, backend policy, release
  audit, and specification documents in navigation.
- Keep site content consistent with current `0.5.0` project state and append-only release history.
- Reserve structure for future English/other languages without generating English placeholders now.

## 3. Technical Choices

- Documentation framework: MkDocs
- Theme: Material for MkDocs
- Default language: `zh-Hans`
- Markdown extensions: `admonition`, `tables`, `toc`, `pymdownx.highlight`,
  `pymdownx.superfences`
- Deployment: GitHub Pages

Documentation dependencies are written to `docs-site/requirements.txt`, not added
to root-project runtime dependencies.

## 4. Directory Structure

The following structure must exist:

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

`docs-site/specs/` contains site-construction PRDs; `docs-site/docs/zh-Hans/specs/`
uses symlinks to existing root `specs/` project specifications.

## 5. Content Source Rules

### 5.1 Existing Files That Must Use Symlinks

Use symlinks when bringing these existing files into the documentation site:

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

Do not copy these files to `docs-site/docs/`.

### 5.2 Files That May Be Added

Only site-project files may be added:

- `docs-site/mkdocs.yml`
- `docs-site/requirements.txt`
- `docs-site/docs/zh-Hans/index.md`
- `docs-site/docs/zh-Hans/getting-started/*.md`
- `docs-site/docs/zh-Hans/user-guide/*.md`
- `docs-site/specs/*.zh-Hans.md`

New pages should be concise and primarily be navigation entries, without duplicating
existing long documents.

## 6. Navigation Requirements

Navigation follows current project capability and contains at least:

- Home
- Quick Start
- Installation and Environment Diagnostics
- Backend Selection Policy
- Estimator API
- Release overview and versioned Release Audits
- Changelog
- Project Specifications
- Documentation Site PRD

Navigation references existing files through their symlink paths.

## 7. Current Project-State Requirements

Documentation must reflect `0.5.0`:

- `mpsboost==0.5.0` is the current release target.
- Apple Silicon wheels support `cp313` / `macosx_13_0_arm64`.
- MPSBoost native CPU/MPS backends remain the core implementation.
- The CPU backend is the correctness oracle and is not replaced by S22 portable-backend planning.
- Delivered model families include GBDT regression/classification, native CPU multiclass
  softmax, DecisionTree, RandomForest, ExtraTrees, CatBoost-like numeric estimators,
  IsolationForest, and LearningToRankRegressor.
- Isolation/ranking are CPU-suitable workflows; `device="mps"` should say CPU is
  more suitable and continue running.
- Missing environments provide copy-ready installation commands and the
  `MPSBOOST_SKIP_ENV_CHECK=1` bypass.
- S22 portable-backend diagnostics and policy are delivered as explicit optional adapter guidance, not hidden native replacement.

## 8. MkDocs Configuration Requirements

`docs-site/mkdocs.yml` must:

- set `site_name: MPSBoost`;
- set the Chinese site language;
- use the Material theme;
- explicitly set `docs_dir: docs`;
- explicitly set `site_dir: site`;
- configure navigation;
- enable base Markdown extensions required for strict builds;
- not reference nonexistent pages.

## 9. Local Validation

Recommended local commands:

```bash
python -m pip install -r docs-site/requirements.txt
mkdocs build --config-file docs-site/mkdocs.yml --strict
```

When the execution environment cannot install dependencies, at minimum inspect
symlinks, configuration files, and navigation-path consistency.

## 10. Acceptance Criteria

Completion requires:

1. Both site PRDs are `.zh-Hans.md` files.
2. The `docs-site/` directory skeleton is complete.
3. MkDocs configuration exists and matches the current MPSBoost project.
4. Existing project documentation enters through symlinks.
5. No source README, CHANGELOG, RELEASE_AUDIT, root `specs/`, or similar files are copied.
6. Chinese home and basic entry pages exist.
7. Local `mkdocs build --strict` passes, or inability to run due to missing
   dependencies is explicitly recorded.
8. The GitHub Pages workflow can build with this site configuration.

## 11. Execution Instructions

Implementers must directly:

- rename original `github_action_prd.md` and `mkdocs_prd.md` as semantic
  `.zh-Hans.md` files;
- establish the `docs-site` directory skeleton;
- add MkDocs configuration and dependency files;
- bring existing documentation in via symlinks;
- add the minimum necessary Chinese entry pages;
- do not address S20 program-file Chinese cleanup.

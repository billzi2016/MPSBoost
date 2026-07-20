# Implementation Task List

> **Translation status:** the complete source task list currently lives in `tasks.zh-Hans.md`. This English file is the in-place English counterpart and must be expanded by S23 without shortening, summarizing, deleting sections, merging bullets, or changing code blocks, commands, constraints, acceptance criteria, or meaning.

> **Translation discipline:** Terra or any low-cost translation agent must preserve one-to-one structure with `tasks.zh-Hans.md`. If a section is not translated yet, keep an explicit placeholder for that section instead of silently omitting it.

## S23: Documentation Site Translation and Internationalization

- [ ] S23.1 Inventory every Markdown source that must appear in the documentation site. Project documentation and project specs must be translated in place beside their source files, for example `README.md` with `README.zh-Hans.md`, and `specs/tasks.md` with `specs/tasks.zh-Hans.md`.
- [ ] S23.2 Keep docs-site-specific PRD source files in `docs-site/specs/`, with English `*.md` and Simplified Chinese `*.zh-Hans.md` files side by side. These files must not be moved into the root `specs/` directory.
- [ ] S23.3 Keep `docs-site/docs/en/` and `docs-site/docs/zh-Hans/` as language navigation trees made from symlinks whenever the source file already exists elsewhere. Do not copy existing project Markdown into `docs-site/docs/`.
- [ ] S23.4 Use `docs-site/docs/en/docs-site-prd/` and `docs-site/docs/zh-Hans/docs-site-prd/` for docs-site PRD navigation links, both pointing back to `docs-site/specs/`.
- [ ] S23.5 Translate README, CHANGELOG, RELEASE_AUDIT, mps_boost_skill, core specs, docs-site PRDs, benchmark docs, test docs, and user-facing guide pages. Translation must happen in the owning source directory, not by maintaining a separate duplicate translation tree under `docs-site/`.
- [ ] S23.6 Add and maintain MkDocs i18n configuration with parallel `en/` and `zh-Hans/` navigation structures. Navigation paths must never point from English pages to Chinese filenames or from Chinese pages to English-only content unless the page is explicitly marked untranslated.
- [ ] S23.7 Validate all bilingual page links, terminology, version numbers, backend policy statements, PyPI install commands, and environment diagnostic commands.
- [ ] S23.8 Translation discipline: do not shorten, summarize, delete sections, merge bullets, simplify warnings, remove caveats, or replace concrete commands with vague prose. English and Chinese pages must preserve the same section structure, information density, code blocks, commands, constraints, limitations, and acceptance criteria.
- [ ] S23.9 Acceptance G21: the documentation site has a maintainable bilingual matrix, every existing Markdown source has an intentional bilingual path, symlinks are valid, MkDocs strict build passes, and translation work does not change the released source semantics for `0.3.0`.

# Legacy Issues

This file records known cleanup work that is important for release quality but
must not be mixed into unrelated feature implementation. Future agents must read
this file before starting the corresponding task in `specs/tasks.md`.

## Program Files Must Use English Only

MPSBoost v2 and later program files must not contain Chinese text. This includes
source code, tests, build scripts, package metadata, generated code templates,
runtime error messages, docstrings, inline comments, and test names.

Documentation files are exempt from this cleanup rule. Chinese text is allowed in
`README` files, `specs/`, project planning documents, and future bilingual site
content. The goal is to make installed code, logs, exceptions, and developer APIs
consistent for international users while preserving Chinese project planning
materials.

## Search Scope

Agents must search program files only. Do not treat Chinese text in `specs/`,
`README.md`, `README.zh-*`, markdown documentation, or website content as a
failure for this task.

Use this command to find Chinese characters in likely program files:

```bash
rg -n "[\u4e00-\u9fff]" \
  src tests scripts include CMakeLists.txt pyproject.toml \
  -g '*.py' -g '*.cpp' -g '*.hpp' -g '*.h' -g '*.mm' -g '*.metal' \
  -g '*.cmake' -g 'CMakeLists.txt' -g '*.toml' -g '*.sh' -g '*.yml' -g '*.yaml'
```

If new program directories are added later, extend the command instead of
manually checking files one by one.

## Required Handling

- Translate user-facing runtime errors to clear English.
- Translate code comments to concise English, keeping only comments that help
  maintainers understand non-obvious logic.
- Translate docstrings, test descriptions, and assertion messages to English.
- Preserve behavior exactly; this task is text cleanup, not a logic refactor.
- Do not weaken tests, skip tests, delete assertions, or change expected numeric
  behavior to make the cleanup pass.
- Do not edit documentation merely because it contains Chinese.

## Completion Standard

This task can be marked complete only when:

- The search command above returns no Chinese text in program files.
- Any additional program directories added since this document was written have
  also been searched.
- A focused test set covering touched modules has passed.
- The final commit contains only translation/comment cleanup unless an explicitly
  approved build fix is required.

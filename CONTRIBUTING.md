# Contributing

MPSBoost is not open for general external contributions yet.

The blocker is validation, not interest. This project depends on real Apple
Silicon MPS/Metal behavior, including native kernels, command synchronization,
model determinism, and CPU/MPS parity. Standard hosted GitHub CI does not provide
the required MPS hardware environment, so a normal pull-request workflow cannot
currently prove that a change is correct.

Self-hosted MPS CI also has a hard security problem: it is very difficult to
fully defend a personal machine that executes external code. I am still studying
the right security model for this project. Until that is solved, general code
contributions are not enabled.

Until the project has a reliable self-hosted MPS CI path and a stable review
process, contributions are limited to maintainer-controlled development.

Issues, benchmark reports, reproducible bugs, and design discussions are still
welcome. The maintainer will review them and make project changes when the report
is actionable. Code contributions will be opened later when the validation
pipeline can protect the project quality bar.

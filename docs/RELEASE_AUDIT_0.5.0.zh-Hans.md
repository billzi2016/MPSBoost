# MPSBoost 0.5.0 发布审计

`0.5.0` 是 0.x line 的 zero-known-blocking-issue 加固版本。它不替代 `0.4.0` 大规模验证版本；它在未来任何 `1.0.0` 承诺之前补上客户侧 fallback 行为和 known-issue audit。

## 范围

`0.5.0` 包含 `0.4.0` 范围，并额外包含：

- portable backend fallback 加固：用户显式选择 external policy 时，如果 external runtime 没有对当前 estimator 激活，则 warning 并通过 native CPU compatibility 保持 workflow 可运行；
- training summary 中记录 requested/effective portable backend；
- versioned known-issue audit；
- 带历史 baseline caveat 和 `1.0.0` 必需审计范围的 versioned performance report；
- 到 `0.5.0` 的 append-only release documentation。

## 发布前必须验证

- portable backend unit tests；
- diagnostics tests；
- packaging public API tests；
- real-world opt-in skip matrix；
- MkDocs strict build 和 symlink check；
- wheel build、`twine check` 和 fresh wheel install smoke verification；
- push 后 GitHub CI 和 Docs success。

性能边界见 `docs/PERFORMANCE_0.5.0.zh-Hans.md`。历史 S4/S6 结果保留为 v2/v3 前 baseline，不作为 current-HEAD 最终 speed claim。

## Artifact Candidate

- Wheel：`dist/mpsboost-0.5.0-cp313-cp313-macosx_26_0_arm64.whl`
- Size：284K
- SHA-256：`6eaef0e1f5620b930036ba8e44dbe8bd70e5dc149ece91dea474e961e671b008`
- GitHub CI：`29981349617`，success
- GitHub Docs：`29981349586`，success
- `twine check`：passed
- Fresh wheel smoke：version `0.5.0`、import、CPU training 和 diagnostics passed
- PyPI URL：https://pypi.org/project/mpsboost/0.5.0/
- Fresh PyPI install smoke：version `0.5.0`、import、CPU training、`system_info()` 和 optional dependency diagnostics passed

## 1.0 边界

`1.0.0` 仍然被阻塞，直到所有计划中的客户侧失败路径、完整真实世界矩阵门槛、性能/内存/权限审计、artifact hash 和用户明确最终确认全部完成。

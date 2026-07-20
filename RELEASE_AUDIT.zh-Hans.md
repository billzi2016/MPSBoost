# MPSBoost 0.3.0 发布审计

本文件记录 `0.3.0` v2 arboretum 里程碑的发布门。

## 范围

`0.3.0` 支持：

- dense numeric regression；
- 二分类和多分类 classification；
- squared-error objective；
- quantile、Poisson 和 Tweedie 回归目标；
- 确定性 quantization；
- 深度受限 histogram tree；
- decision tree、random forest、ExtraTrees 和 CatBoost-like numeric estimator；
- native CPU 多分类 softmax，并保留显式 OvR compatibility；
- 适合 CPU 的 isolation forest anomaly scoring；
- 带 query-group validation 的 pointwise learning-to-rank scoring；
- 真实 MPS gradient、histogram、split-scan、partition 和 buffer-pool path；
- 显式 CPU oracle mode；
- model save/load；
- feature importance、permutation importance 和受控 SHAP-like explanation；
- 导入期 MPS 环境提示，并给出可复制 setup 和 skip 命令；
- cache diagnostics、显式 cache creation 和安全 cache clearing。

不包含：

- sparse matrix；
- native MPS 多分类 softmax；
- 官方 third-party SHAP TreeExplainer integration；
- categorical model persistence；
- public GPU prediction；
- 完整 third-party API compatibility。

## License

- 项目许可证：Apache-2.0。
- Runtime dependency：NumPy，许可表达式由 package metadata 报告，为 permissive license。
- Build/test-only dependency 不会打包进 runtime wheel。
- Wheel 必须包含项目 `LICENSE` 文件。

## Wheel 内容规则

Release wheel 只能包含 runtime package file：

- Python package file；
- native extension；
- 编译后的 Metal shader library；
- package metadata 和 license metadata。

Release wheel 不得包含：

- `specs/`；
- `tests/`；
- `benchmarks/`；
- `.github/`；
- build directory；
- cache directory；
- raw `.metal`、`.air` 或 temporary shader files；
- credential 或 runner file。

## Dynamic Link 规则

Native extension 可以链接 Python、C++、Objective-C runtime、Foundation、CoreFoundation 和 Metal 所需的 macOS system library 与 framework。它不得链接 heavyweight ML runtime，也不得链接 private project-local absolute path。

## Validation Matrix

发布 `0.3.0` 前必须完成：

- 本地 full test suite；
- Python 3.10 和 3.13 上的 GitHub hosted CPU/package test；
- Python 3.10 和 3.13 上的 self-hosted real Metal GPU test；
- exact uploaded wheels 的 `twine check`；
- fresh PyPI install 和 real MPS smoke test。

## Benchmark Evidence

已提交的 benchmark result：

- `benchmarks/results/s4-m2-ultra-py313.json`
- `benchmarks/results/s4-m2-ultra-py313.md`
- `benchmarks/results/s6-m2-ultra-py313.json`
- `benchmarks/results/s6-m2-ultra-py313.md`

S6 report 同时记录 GPU 胜出场景和 small-data regression。`gbdt-large-wide` 端到端场景在记录的 M2 Ultra 验证机器上达到 1.629x median speedup。

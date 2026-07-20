# 贡献指南

MPSBoost 目前尚未开放普通外部贡献。

阻塞点是验证能力，而不是兴趣。本项目依赖真实 Apple Silicon MPS/Metal 行为，包括 native kernel、command synchronization、model determinism 和 CPU/MPS parity。标准 hosted GitHub CI 不提供所需的 MPS 硬件环境，因此普通 pull-request workflow 目前无法证明变更是正确的。

Self-hosted MPS CI 也有明确安全问题：让个人机器执行外部代码很难做到完全防护。我仍在研究本项目合适的安全模型。在这个问题解决之前，普通代码贡献暂不启用。

在项目拥有可靠 self-hosted MPS CI 路径和稳定 review 流程之前，贡献范围限制为 maintainer-controlled development。

Issue、benchmark report、可复现 bug 和设计讨论仍然欢迎。维护者会 review 这些内容，并在报告可操作时修改项目。等验证 pipeline 能够保护项目质量门后，再开放代码贡献。

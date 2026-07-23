# 安装与环境诊断

## 安装

```bash
python -m pip install mpsboost
```

当前 `0.5.0` PyPI wheel 面向 Apple Silicon macOS。CPU 训练可直接使用；MPS 加速需要 Apple Silicon、Metal 和可用的本机环境。

## MPS 环境缺失时

如果导入或训练时发现 Apple GPU 加速不可用，MPSBoost 会给出可复制命令：

```bash
xcode-select --install
xcodebuild -downloadComponent MetalToolchain
python -m pip install --upgrade --force-reinstall mpsboost
python -c "import mpsboost as mb; print(mb.system_info())"
```

CPU-only 任务、CI 或 `GridSearchCV` worker 可以跳过导入期环境检查：

```bash
MPSBOOST_SKIP_ENV_CHECK=1 python your_script.py
```

跳过环境检查不会禁用 CPU 训练。用户强制 `device="mps"` 但环境不可用时，错误信息会继续包含上述修复和跳过命令。

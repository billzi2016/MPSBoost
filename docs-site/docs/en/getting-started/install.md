# Installation and Environment

## Install

```bash
python -m pip install mpsboost
```

The `0.4.0` PyPI wheel targets Apple Silicon macOS. CPU training works directly.
MPS acceleration requires Apple Silicon, Metal, and a usable local environment.

## When MPS Is Unavailable

```bash
xcode-select --install
xcodebuild -downloadComponent MetalToolchain
python -m pip install --upgrade --force-reinstall mpsboost
python -c "import mpsboost as mb; print(mb.system_info())"
```

CPU-only jobs, CI, and `GridSearchCV` workers can skip the import-time check:

```bash
MPSBOOST_SKIP_ENV_CHECK=1 python your_script.py
```

Skipping this check does not disable CPU training.

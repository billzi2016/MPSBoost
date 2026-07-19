"""MPSBoost 后端诊断与真实 GPU smoke 验证。

本模块负责定位 wheel 内的 shader 资源，并把原生层的非敏感设备信息整理为稳定 Python
字典。它不创建缓存、不访问网络、不初始化训练状态，也不把测试入口导出到顶层 API。
"""

from __future__ import annotations

from contextlib import contextmanager
from importlib.resources import as_file, files
from typing import Any, Iterator, Sequence

from . import _native


def is_available() -> bool:
    """返回当前 wheel 与机器是否具备可使用的真实 MPS 后端。

    该结果来自原生 Metal 设备查询，不根据平台字符串猜测。查询无文件写入和网络副作用；
    原生扩展或 shader 缺失属于安装损坏，会在导入或 smoke 验证时明确失败。
    """

    return bool(_native._backend_info()["available"])


def system_info() -> dict[str, Any]:
    """返回可用于安装诊断的非敏感后端信息。

    返回值不包含设备持久标识、用户名、路径或环境变量。字节大小使用整数返回，展示单位
    由调用方决定，避免诊断层引入格式化语义。
    """

    info = dict(_native._backend_info())
    info.update(
        {
            "backend": "mps",
            "package_version": _native.__version__,
        }
    )
    return info


@contextmanager
def _metallib_path() -> Iterator[str]:
    """在上下文存续期内返回 wheel 中唯一 shader library 的真实路径。

    estimator、kernel 测试和 smoke 共享此资源定位逻辑，避免不同入口对压缩资源或临时
    文件生命周期作出不同假设。调用方必须在 ``with`` 块内完成所有 GPU 工作。
    """

    resource = files("mpsboost").joinpath("_kernels.metallib")
    with as_file(resource) as metallib_path:
        yield str(metallib_path)


def _run_vector_add_for_test(
    left: Sequence[float], right: Sequence[float]
) -> list[float]:
    """在真实 GPU 上执行内部向量加法集成测试。

    Args:
        left: 左输入序列。
        right: 与左输入等长的右输入序列。

    Returns:
        GPU 计算得到的逐元素和。

    Raises:
        BackendError: 设备、shader、pipeline、buffer 或 command 任一阶段失败。

    ``as_file`` 同时支持普通 wheel 文件和未来可能的压缩资源。原生调用必须在上下文内
    完成，防止临时资源路径在 GPU 加载前失效。
    """

    with _metallib_path() as metallib_path:
        return list(_native._vector_add(left, right, metallib_path))

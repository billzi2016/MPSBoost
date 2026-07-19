// MPSBoost Python 原生绑定。
//
// 职责：把稳定 C++ POD 和异常转换为轻量 Python 对象。这里不得复制设备实现、参数
// 校验或训练算法。所有公开用户体验由 Python 层组织，以下下划线入口属于内部接口。

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mpsboost/backend.hpp"
#include "mpsboost/version.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_native, module) {
  module.doc() = "MPSBoost 原生 MPS/Metal 后端";
  module.attr("__version__") = MPSBOOST_VERSION;

  // 统一注册后端异常，使 Python 可以明确区分设备执行失败与普通参数错误。
  py::register_exception<mpsboost::BackendError>(module, "BackendError");

  module.def(
      "_backend_info",
      []() {
        const mpsboost::BackendInfo info = mpsboost::QueryBackendInfo();
        py::dict result;
        result["available"] = info.available;
        result["device_name"] = info.available ? py::cast(info.device_name) : py::none();
        result["recommended_max_working_set_size"] =
            info.recommended_max_working_set_size;
        result["has_unified_memory"] = info.has_unified_memory;
        return result;
      },
      "返回不包含敏感标识的真实 Metal 设备能力摘要。");

  module.def(
      "_vector_add",
      &mpsboost::RunVectorAdd,
      py::arg("left"),
      py::arg("right"),
      py::arg("metallib_path"),
      "在真实 GPU 上执行 smoke 向量加法，仅用于后端集成验证。");
}

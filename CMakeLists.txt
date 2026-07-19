# MPSBoost 原生构建入口。
#
# 职责：构建 Python 扩展、编译 Metal shader、安装运行时资源。业务算法和用户参数
# 不得写入构建脚本。普通用户安装预编译 wheel 时不会执行此文件。
cmake_minimum_required(VERSION 3.25)

# 默认面向仍受项目支持的较低 macOS 版本，避免在新系统构建出的 wheel 被错误限制为
# 只能在构建机系统运行。发布流水线仍可显式传入更高目标进行兼容性验证。
if(NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "MPSBoost 最低 macOS 版本")
endif()

project(MPSBoost LANGUAGES CXX OBJCXX)

if(NOT APPLE)
  message(FATAL_ERROR "MPSBoost native backend currently requires macOS")
endif()

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
  message(FATAL_ERROR "MPSBoost native backend currently requires Apple Silicon")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_program(MPSBOOST_XCRUN_EXECUTABLE xcrun REQUIRED)
find_library(MPSBOOST_FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(MPSBOOST_METAL_FRAMEWORK Metal REQUIRED)

# scikit-build-core 从 pyproject.toml 注入版本。直接运行 CMake 时使用开发后备值，
# 但正式 wheel 的唯一版本来源始终是 pyproject.toml。
if(DEFINED SKBUILD_PROJECT_VERSION_FULL)
  # FULL 保留 a1、rc1 等预发布后缀；非 FULL 变量只含基础三段版本。
  set(MPSBOOST_VERSION "${SKBUILD_PROJECT_VERSION_FULL}")
else()
  set(MPSBOOST_VERSION "0.2.0")
endif()

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/mpsboost/version.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/mpsboost/version.hpp"
  @ONLY
)

set(MPSBOOST_METAL_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/mps/kernels/smoke.metal"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/mps/kernels/gradients.metal"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/mps/kernels/histogram.metal"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/mps/kernels/split_scan.metal"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/mps/kernels/partition.metal"
)
set(MPSBOOST_METALLIB_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/_kernels.metallib")

# 每个 shader 独立生成 AIR，再链接为唯一 metallib。这样增量构建能准确失效单个源，
# 最终用户仍只加载一个版本匹配的资源，不需要 Xcode 或本地 shader 编译器。
set(MPSBOOST_AIR_OUTPUTS)
foreach(MPSBOOST_METAL_SOURCE IN LISTS MPSBOOST_METAL_SOURCES)
  get_filename_component(MPSBOOST_METAL_STEM "${MPSBOOST_METAL_SOURCE}" NAME_WE)
  set(MPSBOOST_AIR_OUTPUT
      "${CMAKE_CURRENT_BINARY_DIR}/mpsboost_${MPSBOOST_METAL_STEM}.air")
  add_custom_command(
    OUTPUT "${MPSBOOST_AIR_OUTPUT}"
    COMMAND "${MPSBOOST_XCRUN_EXECUTABLE}" -sdk macosx metal
            -c "${MPSBOOST_METAL_SOURCE}" -o "${MPSBOOST_AIR_OUTPUT}"
    DEPENDS "${MPSBOOST_METAL_SOURCE}"
    COMMENT "编译 MPSBoost Metal shader: ${MPSBOOST_METAL_STEM}"
    VERBATIM
  )
  list(APPEND MPSBOOST_AIR_OUTPUTS "${MPSBOOST_AIR_OUTPUT}")
endforeach()

add_custom_command(
  OUTPUT "${MPSBOOST_METALLIB_OUTPUT}"
  COMMAND "${MPSBOOST_XCRUN_EXECUTABLE}" -sdk macosx metallib
          ${MPSBOOST_AIR_OUTPUTS} -o "${MPSBOOST_METALLIB_OUTPUT}"
  DEPENDS ${MPSBOOST_AIR_OUTPUTS}
  COMMENT "编译 MPSBoost Metal shader library"
  VERBATIM
)
add_custom_target(mpsboost_metallib ALL DEPENDS "${MPSBOOST_METALLIB_OUTPUT}")

pybind11_add_module(
  _native MODULE
  src/python/bindings.cpp
  src/python/binding_backend.cpp
  src/python/binding_dataset.cpp
  src/python/binding_dataset_tools.cpp
  src/python/binding_helpers.cpp
  src/python/binding_model.cpp
  src/python/binding_objective.cpp
  src/core/binned_dataset.cpp
  src/core/binned_dataset_internal.cpp
  src/core/binned_dataset_serialization.cpp
  src/core/objective.cpp
  src/core/tree.cpp
  src/core/tree_histogram.cpp
  src/core/tree_leafwise.cpp
  src/core/tree_split.cpp
  src/core/tree_structure.cpp
  src/core/trainer.cpp
  src/cpu/reference_backend.cpp
  src/io/model_file_io.cpp
  src/io/model_format.cpp
  src/io/model_payload.cpp
  src/mps/metal_context.mm
  src/mps/mps_backend.mm
  src/mps/mps_context.mm
  src/mps/mps_gradients.mm
  src/mps/mps_histogram.mm
  src/mps/mps_split_partition.mm
)
add_dependencies(_native mpsboost_metallib)

target_include_directories(
  _native PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_BINARY_DIR}/generated"
)
target_link_libraries(
  _native PRIVATE
  "${MPSBOOST_FOUNDATION_FRAMEWORK}"
  "${MPSBOOST_METAL_FRAMEWORK}"
)
target_compile_options(_native PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Werror>
  $<$<COMPILE_LANGUAGE:OBJCXX>:-Wall;-Wextra;-Wpedantic;-Werror;-fobjc-arc>
)

install(TARGETS _native LIBRARY DESTINATION mpsboost)
install(FILES "${MPSBOOST_METALLIB_OUTPUT}" DESTINATION mpsboost)

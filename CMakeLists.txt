# MPSBoost native build entry.
#
# Responsibility: builds the Python extension, compiles Metal shaders, and installs
# runtime resources. Business algorithms and user parameters do not belong here.
# This file does not run when ordinary users install a prebuilt wheel.
cmake_minimum_required(VERSION 3.25)

# Default to the oldest macOS version still supported, preventing wheels built on
# newer systems from being incorrectly restricted to the build host. Release jobs
# may explicitly pass a higher target for compatibility validation.
if(NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "MPSBoost minimum macOS version")
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

# scikit-build-core injects the version from pyproject.toml. Direct CMake uses a
# development fallback, but pyproject.toml remains the sole version source for wheels.
if(DEFINED SKBUILD_PROJECT_VERSION_FULL)
  # FULL retains prerelease suffixes such as a1 and rc1; the non-FULL variable has
  # only the base three-part version.
  set(MPSBOOST_VERSION "${SKBUILD_PROJECT_VERSION_FULL}")
else()
  set(MPSBOOST_VERSION "0.4.0")
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

# Each shader produces independent AIR, then links into one metallib. Incremental
# builds invalidate individual sources precisely while users load one matching
# resource without Xcode or a local shader compiler.
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
    COMMENT "Compile MPSBoost Metal shader: ${MPSBOOST_METAL_STEM}"
    VERBATIM
  )
  list(APPEND MPSBOOST_AIR_OUTPUTS "${MPSBOOST_AIR_OUTPUT}")
endforeach()

add_custom_command(
  OUTPUT "${MPSBOOST_METALLIB_OUTPUT}"
  COMMAND "${MPSBOOST_XCRUN_EXECUTABLE}" -sdk macosx metallib
          ${MPSBOOST_AIR_OUTPUTS} -o "${MPSBOOST_METALLIB_OUTPUT}"
  DEPENDS ${MPSBOOST_AIR_OUTPUTS}
  COMMENT "Compile MPSBoost Metal shader library"
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
  src/core/tree_interactions.cpp
  src/core/tree_leafwise.cpp
  src/core/tree_monotonic.cpp
  src/core/tree_split.cpp
  src/core/tree_structure.cpp
  src/core/multiclass_trainer.cpp
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

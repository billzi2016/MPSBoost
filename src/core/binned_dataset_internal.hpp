// Internal helpers shared by binned dataset implementation files.
//
// This header exposes only validation and overflow utilities needed to keep quantization and
// serialization split without duplicating safety checks.
#pragma once

#include <cstddef>
#include <cstdint>

#include "mpsboost/binned_dataset.hpp"

namespace mpsboost::internal {

std::uint64_t CheckedMultiply(std::uint64_t left,
                              std::uint64_t right,
                              const char* context);
std::uint64_t CheckedAdd(std::uint64_t left,
                         std::uint64_t right,
                         const char* context);
std::size_t CheckedSize(std::uint64_t value, const char* context);
std::uint64_t ScalarByteSize(ScalarType type);
void ValidateSchemaFields(const QuantizationSchema& schema);

}  // namespace mpsboost::internal

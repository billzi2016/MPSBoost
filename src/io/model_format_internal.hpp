// Private binary-format contracts shared by model serialization units.
//
// This header centralizes byte order, finite-value validation, and format
// constants so framing and payload code cannot develop incompatible codecs.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "mpsboost/trainer.hpp"

namespace mpsboost::model_format_internal {

inline constexpr std::array<std::uint8_t, 8> kMagic{
    'M', 'P', 'S', 'B', 'M', 'O', 'D', 0};
inline constexpr std::uint16_t kFormatMajor = 1;
inline constexpr std::uint16_t kFormatMinor = 1;
inline constexpr std::size_t kHeaderSize = 32;

template <typename Integer>
void AppendUnsigned(std::vector<std::uint8_t>* output, Integer value) {
  static_assert(std::is_unsigned_v<Integer>);
  for (std::size_t index = 0; index < sizeof(Integer); ++index) {
    output->push_back(
        static_cast<std::uint8_t>((value >> (index * 8U)) & 0xFFU));
  }
}

template <typename Float, typename Bits>
void AppendFloat(std::vector<std::uint8_t>* output, Float value) {
  static_assert(sizeof(Float) == sizeof(Bits));
  Bits bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  AppendUnsigned(output, bits);
}

class Reader final {
 public:
  Reader(const std::uint8_t* data, std::size_t size)
      : data_(data), size_(size) {}

  template <typename Integer>
  Integer ReadUnsigned(const char* field) {
    static_assert(std::is_unsigned_v<Integer>);
    Require(sizeof(Integer), field);
    Integer value = 0;
    for (std::size_t index = 0; index < sizeof(Integer); ++index) {
      value |= static_cast<Integer>(data_[position_++]) << (index * 8U);
    }
    return value;
  }

  template <typename Float, typename Bits>
  Float ReadFloat(const char* field) {
    const Bits bits = ReadUnsigned<Bits>(field);
    Float value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    if (!std::isfinite(value)) {
      throw TrainingError(std::string("模型浮点字段不是有限值：") + field);
    }
    return value;
  }

  bool at_end() const noexcept { return position_ == size_; }

 private:
  void Require(std::size_t count, const char* field) {
    if (count > size_ - position_) {
      throw TrainingError(std::string("模型数据截断：") + field);
    }
  }

  const std::uint8_t* data_;
  std::size_t size_;
  std::size_t position_{0};
};

std::uint64_t Checksum(const std::uint8_t* data, std::size_t size);

std::vector<std::uint8_t> BuildPayload(const RegressionModel& model);

RegressionModel ParsePayload(const std::uint8_t* data,
                             std::size_t size,
                             std::uint16_t format_minor);

}  // namespace mpsboost::model_format_internal

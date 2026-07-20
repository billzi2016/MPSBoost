// Versioned container framing for native MPSBoost model bytes.
//
// This unit owns the magic, version header, payload length, and checksum.
// Payload semantics and filesystem persistence remain separate responsibilities.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "mpsboost/trainer.hpp"
#include "model_format_internal.hpp"

namespace mpsboost::model_format_internal {
namespace {

constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

}  // namespace

std::uint64_t Checksum(const std::uint8_t* data, std::size_t size) {
  std::uint64_t result = kFnvOffset;
  for (std::size_t index = 0; index < size; ++index) {
    result = (result ^ data[index]) * kFnvPrime;
  }
  return result;
}

}  // namespace mpsboost::model_format_internal

namespace mpsboost {
using model_format_internal::AppendUnsigned;
using model_format_internal::BuildPayload;
using model_format_internal::Checksum;
using model_format_internal::kFormatMajor;
using model_format_internal::kFormatMinor;
using model_format_internal::kHeaderSize;
using model_format_internal::kMagic;
using model_format_internal::kMulticlassModelKind;
using model_format_internal::kRegressionModelKind;
using model_format_internal::ParseMulticlassPayload;
using model_format_internal::ParsePayload;
using model_format_internal::Reader;

std::vector<std::uint8_t> SerializeModel(const RegressionModel& model) {
  const std::vector<std::uint8_t> payload = BuildPayload(model);
  std::vector<std::uint8_t> output(kMagic.begin(), kMagic.end());
  AppendUnsigned(&output, kFormatMajor);
  AppendUnsigned(&output, kFormatMinor);
  AppendUnsigned(&output, kRegressionModelKind);
  AppendUnsigned(&output, static_cast<std::uint64_t>(payload.size()));
  AppendUnsigned(&output, Checksum(payload.data(), payload.size()));
  output.insert(output.end(), payload.begin(), payload.end());
  return output;
}

std::vector<std::uint8_t> SerializeModel(const MulticlassModel& model) {
  const std::vector<std::uint8_t> payload = BuildPayload(model);
  std::vector<std::uint8_t> output(kMagic.begin(), kMagic.end());
  AppendUnsigned(&output, kFormatMajor);
  AppendUnsigned(&output, kFormatMinor);
  AppendUnsigned(&output, kMulticlassModelKind);
  AppendUnsigned(&output, static_cast<std::uint64_t>(payload.size()));
  AppendUnsigned(&output, Checksum(payload.data(), payload.size()));
  output.insert(output.end(), payload.begin(), payload.end());
  return output;
}

RegressionModel DeserializeModel(const std::vector<std::uint8_t>& bytes) {
  if (bytes.size() < kHeaderSize ||
      !std::equal(kMagic.begin(), kMagic.end(), bytes.begin())) {
    throw TrainingError("模型 magic 不匹配或头部截断");
  }
  Reader header(bytes.data() + kMagic.size(), bytes.size() - kMagic.size());
  const std::uint16_t major = header.ReadUnsigned<std::uint16_t>("major");
  const std::uint16_t minor = header.ReadUnsigned<std::uint16_t>("minor");
  const std::uint32_t model_kind = header.ReadUnsigned<std::uint32_t>("model kind");
  const std::uint64_t payload_size = header.ReadUnsigned<std::uint64_t>("payload size");
  const std::uint64_t checksum = header.ReadUnsigned<std::uint64_t>("checksum");
  if (major != kFormatMajor) {
    throw TrainingError("模型 major 版本不受支持");
  }
  if (model_kind != kRegressionModelKind) {
    throw TrainingError("model kind is incompatible with regression loader");
  }
  if (payload_size != bytes.size() - kHeaderSize) {
    throw TrainingError("模型 payload 长度不一致");
  }
  const std::uint8_t* payload = bytes.data() + kHeaderSize;
  if (Checksum(payload, static_cast<std::size_t>(payload_size)) != checksum) {
    throw TrainingError("模型完整性校验失败");
  }
  return ParsePayload(payload, static_cast<std::size_t>(payload_size), minor);
}

MulticlassModel DeserializeMulticlassModel(const std::vector<std::uint8_t>& bytes) {
  if (bytes.size() < kHeaderSize ||
      !std::equal(kMagic.begin(), kMagic.end(), bytes.begin())) {
    throw TrainingError("模型 magic 不匹配或头部截断");
  }
  Reader header(bytes.data() + kMagic.size(), bytes.size() - kMagic.size());
  const std::uint16_t major = header.ReadUnsigned<std::uint16_t>("major");
  const std::uint16_t minor = header.ReadUnsigned<std::uint16_t>("minor");
  const std::uint32_t model_kind = header.ReadUnsigned<std::uint32_t>("model kind");
  const std::uint64_t payload_size = header.ReadUnsigned<std::uint64_t>("payload size");
  const std::uint64_t checksum = header.ReadUnsigned<std::uint64_t>("checksum");
  if (major != kFormatMajor) {
    throw TrainingError("模型 major 版本不受支持");
  }
  if (model_kind != kMulticlassModelKind) {
    throw TrainingError("model kind is incompatible with multiclass loader");
  }
  if (payload_size != bytes.size() - kHeaderSize) {
    throw TrainingError("模型 payload 长度不一致");
  }
  const std::uint8_t* payload = bytes.data() + kHeaderSize;
  if (Checksum(payload, static_cast<std::size_t>(payload_size)) != checksum) {
    throw TrainingError("模型完整性校验失败");
  }
  return ParseMulticlassPayload(payload, static_cast<std::size_t>(payload_size),
                                minor);
}
}  // namespace mpsboost

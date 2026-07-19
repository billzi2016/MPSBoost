// MPSBoost 版本化模型格式与原子文件 I/O。
//
// 职责：序列化分箱 schema、boosting 元数据和扁平树，加载时检查长度、校验和、索引与
// 数值不变量。格式不保存训练数据或设备信息；本文件不参与训练和预测算法。

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <system_error>
#include <type_traits>

#include "mpsboost/objective.hpp"
#include "mpsboost/trainer.hpp"

namespace mpsboost {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{'M', 'P', 'S', 'B', 'M', 'O', 'D', 0};
constexpr std::uint16_t kFormatMajor = 1;
constexpr std::uint16_t kFormatMinor = 1;
constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

template <typename Integer>
void AppendUnsigned(std::vector<std::uint8_t>* output, Integer value) {
  static_assert(std::is_unsigned_v<Integer>);
  for (std::size_t index = 0; index < sizeof(Integer); ++index) {
    output->push_back(static_cast<std::uint8_t>((value >> (index * 8U)) & 0xFFU));
  }
}

template <typename Float, typename Bits>
void AppendFloat(std::vector<std::uint8_t>* output, Float value) {
  static_assert(sizeof(Float) == sizeof(Bits));
  Bits bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  AppendUnsigned(output, bits);
}

std::uint64_t Checksum(const std::uint8_t* data, std::size_t size) {
  std::uint64_t result = kFnvOffset;
  for (std::size_t index = 0; index < size; ++index) {
    result = (result ^ data[index]) * kFnvPrime;
  }
  return result;
}

class Reader final {
 public:
  Reader(const std::uint8_t* data, std::size_t size) : data_(data), size_(size) {}

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

std::vector<std::uint8_t> BuildPayload(const RegressionModel& model) {
  std::vector<std::uint8_t> payload;
  AppendUnsigned(&payload, model.feature_count());
  AppendUnsigned(&payload, model.schema().max_bins());
  AppendFloat<double, std::uint64_t>(&payload, model.base_score());
  AppendFloat<double, std::uint64_t>(&payload, model.learning_rate());
  AppendUnsigned(
      &payload,
      static_cast<std::uint32_t>(
          model.objective() == TrainingParameters::Objective::kBinaryLogistic ? 1 : 0));
  AppendUnsigned(&payload,
                 static_cast<std::uint64_t>(model.schema().boundaries().size()));
  for (const FeatureBinMetadata& item : model.schema().feature_metadata()) {
    AppendUnsigned(&payload, item.boundary_offset);
    AppendUnsigned(&payload, item.boundary_count);
    AppendUnsigned(&payload, item.bin_count);
  }
  for (const float boundary : model.schema().boundaries()) {
    AppendFloat<float, std::uint32_t>(&payload, boundary);
  }
  AppendUnsigned(&payload, model.tree_count());
  for (const RegressionTree& tree : model.trees()) {
    if (tree.nodes().size() > std::numeric_limits<std::uint32_t>::max()) {
      throw TrainingError("模型树节点数超出格式限制");
    }
    AppendUnsigned(&payload, static_cast<std::uint32_t>(tree.nodes().size()));
    for (const TreeNode& node : tree.nodes()) {
      AppendUnsigned(&payload, node.feature_index);
      AppendUnsigned(&payload, node.threshold_bin);
      AppendUnsigned(&payload, node.left_child);
      AppendUnsigned(&payload, node.right_child);
      AppendFloat<double, std::uint64_t>(&payload, node.leaf_value);
      AppendFloat<double, std::uint64_t>(&payload, node.gain);
      AppendUnsigned(&payload, node.flags);
      for (int index = 0; index < 7; ++index) {
        AppendUnsigned(&payload, std::uint8_t{0});
      }
    }
  }
  return payload;
}

std::string ErrnoMessage(const char* stage) {
  return std::string(stage) + ": " + std::strerror(errno);
}

}  // namespace

std::vector<std::uint8_t> SerializeModel(const RegressionModel& model) {
  const std::vector<std::uint8_t> payload = BuildPayload(model);
  std::vector<std::uint8_t> output(kMagic.begin(), kMagic.end());
  AppendUnsigned(&output, kFormatMajor);
  AppendUnsigned(&output, kFormatMinor);
  AppendUnsigned(&output, static_cast<std::uint32_t>(0));
  AppendUnsigned(&output, static_cast<std::uint64_t>(payload.size()));
  AppendUnsigned(&output, Checksum(payload.data(), payload.size()));
  output.insert(output.end(), payload.begin(), payload.end());
  return output;
}

RegressionModel DeserializeModel(const std::vector<std::uint8_t>& bytes) {
  constexpr std::size_t kHeaderSize = 32;
  if (bytes.size() < kHeaderSize ||
      !std::equal(kMagic.begin(), kMagic.end(), bytes.begin())) {
    throw TrainingError("模型 magic 不匹配或头部截断");
  }
  Reader header(bytes.data() + kMagic.size(), bytes.size() - kMagic.size());
  const std::uint16_t major = header.ReadUnsigned<std::uint16_t>("major");
  const std::uint16_t minor = header.ReadUnsigned<std::uint16_t>("minor");
  static_cast<void>(header.ReadUnsigned<std::uint32_t>("reserved"));
  const std::uint64_t payload_size = header.ReadUnsigned<std::uint64_t>("payload size");
  const std::uint64_t checksum = header.ReadUnsigned<std::uint64_t>("checksum");
  if (major != kFormatMajor) {
    throw TrainingError("模型 major 版本不受支持");
  }
  if (payload_size != bytes.size() - kHeaderSize) {
    throw TrainingError("模型 payload 长度不一致");
  }
  const std::uint8_t* payload = bytes.data() + kHeaderSize;
  if (Checksum(payload, static_cast<std::size_t>(payload_size)) != checksum) {
    throw TrainingError("模型完整性校验失败");
  }

  Reader reader(payload, static_cast<std::size_t>(payload_size));
  const std::uint32_t features = reader.ReadUnsigned<std::uint32_t>("features");
  const std::uint32_t max_bins = reader.ReadUnsigned<std::uint32_t>("max_bins");
  const double base_score = reader.ReadFloat<double, std::uint64_t>("base_score");
  const double learning_rate =
      reader.ReadFloat<double, std::uint64_t>("learning_rate");
  TrainingParameters::Objective objective = TrainingParameters::Objective::kSquaredError;
  if (minor >= 1) {
    const std::uint32_t objective_code =
        reader.ReadUnsigned<std::uint32_t>("objective");
    if (objective_code == 0) {
      objective = TrainingParameters::Objective::kSquaredError;
    } else if (objective_code == 1) {
      objective = TrainingParameters::Objective::kBinaryLogistic;
    } else {
      throw TrainingError("model objective is unsupported");
    }
  }
  const std::uint64_t boundary_count =
      reader.ReadUnsigned<std::uint64_t>("boundary_count");
  if (boundary_count > payload_size / sizeof(float)) {
    throw TrainingError("模型 boundary_count 超出 payload 范围");
  }
  std::vector<FeatureBinMetadata> metadata;
  metadata.reserve(features);
  for (std::uint32_t feature = 0; feature < features; ++feature) {
    metadata.push_back(FeatureBinMetadata{
        reader.ReadUnsigned<std::uint64_t>("boundary_offset"),
        reader.ReadUnsigned<std::uint32_t>("feature boundary_count"),
        reader.ReadUnsigned<std::uint32_t>("feature bin_count")});
  }
  std::vector<float> boundaries;
  boundaries.reserve(static_cast<std::size_t>(boundary_count));
  for (std::uint64_t index = 0; index < boundary_count; ++index) {
    boundaries.push_back(reader.ReadFloat<float, std::uint32_t>("boundary"));
  }
  QuantizationSchema schema = RestoreQuantizationSchema(
      features, max_bins, std::move(boundaries), std::move(metadata));

  const std::uint32_t tree_count = reader.ReadUnsigned<std::uint32_t>("tree_count");
  if (tree_count == 0 || tree_count > payload_size / 41) {
    throw TrainingError("模型树数量不合法");
  }
  std::vector<RegressionTree> trees;
  trees.reserve(tree_count);
  for (std::uint32_t tree = 0; tree < tree_count; ++tree) {
    const std::uint32_t node_count = reader.ReadUnsigned<std::uint32_t>("node_count");
    if (node_count == 0 || node_count > payload_size / 41) {
      throw TrainingError("模型节点数量不合法");
    }
    std::vector<TreeNode> nodes;
    nodes.reserve(node_count);
    for (std::uint32_t node_index = 0; node_index < node_count; ++node_index) {
      TreeNode node;
      node.feature_index = reader.ReadUnsigned<std::uint32_t>("feature_index");
      node.threshold_bin = reader.ReadUnsigned<std::uint32_t>("threshold_bin");
      node.left_child = reader.ReadUnsigned<std::uint32_t>("left_child");
      node.right_child = reader.ReadUnsigned<std::uint32_t>("right_child");
      node.leaf_value = reader.ReadFloat<double, std::uint64_t>("leaf_value");
      node.gain = reader.ReadFloat<double, std::uint64_t>("gain");
      node.flags = reader.ReadUnsigned<std::uint8_t>("flags");
      for (int reserved = 0; reserved < 7; ++reserved) {
        if (reader.ReadUnsigned<std::uint8_t>("node reserved") != 0) {
          throw TrainingError("模型节点保留字段必须为零");
        }
      }
      nodes.push_back(node);
    }
    trees.push_back(RegressionTree::Restore(features, std::move(nodes)));
  }
  if (!reader.at_end()) {
    throw TrainingError("模型包含未识别的尾部字节");
  }
  return RegressionModel::Restore(
      std::move(schema), base_score, learning_rate, objective, std::move(trees));
}

void SaveModelFile(const RegressionModel& model, const std::string& path) {
  if (path.empty()) {
    throw TrainingError("模型保存路径不能为空");
  }
  const std::filesystem::path target(path);
  const std::filesystem::path directory =
      target.parent_path().empty() ? std::filesystem::path(".") : target.parent_path();
  if (!std::filesystem::is_directory(directory)) {
    throw TrainingError("模型保存目录不存在");
  }
  const std::vector<std::uint8_t> bytes = SerializeModel(model);
  std::string temporary_pattern =
      (directory / ("." + target.filename().string() + ".mpsboost-XXXXXX")).string();
  std::vector<char> mutable_pattern(temporary_pattern.begin(), temporary_pattern.end());
  mutable_pattern.push_back('\0');
  const int descriptor = ::mkstemp(mutable_pattern.data());
  if (descriptor < 0) {
    throw TrainingError(ErrnoMessage("创建模型临时文件失败"));
  }
  const std::filesystem::path temporary(mutable_pattern.data());
  try {
    std::size_t written = 0;
    while (written < bytes.size()) {
      const ssize_t count = ::write(descriptor, bytes.data() + written,
                                    bytes.size() - written);
      if (count <= 0) {
        throw TrainingError(ErrnoMessage("写入模型临时文件失败"));
      }
      written += static_cast<std::size_t>(count);
    }
    if (::fsync(descriptor) != 0 || ::close(descriptor) != 0) {
      throw TrainingError(ErrnoMessage("同步模型临时文件失败"));
    }
    if (::rename(temporary.c_str(), target.c_str()) != 0) {
      throw TrainingError(ErrnoMessage("原子替换模型文件失败"));
    }
  } catch (...) {
    ::close(descriptor);
    ::unlink(temporary.c_str());
    throw;
  }
}

RegressionModel LoadModelFile(const std::string& path) {
  if (path.empty()) {
    throw TrainingError("模型加载路径不能为空");
  }
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(path, error);
  if (error || size > std::numeric_limits<std::size_t>::max()) {
    throw TrainingError("无法读取模型文件大小");
  }
  const int descriptor = ::open(path.c_str(), O_RDONLY);
  if (descriptor < 0) {
    throw TrainingError(ErrnoMessage("打开模型文件失败"));
  }
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  std::size_t position = 0;
  while (position < bytes.size()) {
    const ssize_t count = ::read(descriptor, bytes.data() + position,
                                 bytes.size() - position);
    if (count <= 0) {
      ::close(descriptor);
      throw TrainingError(count == 0 ? "模型文件读取提前结束"
                                     : ErrnoMessage("读取模型文件失败"));
    }
    position += static_cast<std::size_t>(count);
  }
  ::close(descriptor);
  return DeserializeModel(bytes);
}

}  // namespace mpsboost

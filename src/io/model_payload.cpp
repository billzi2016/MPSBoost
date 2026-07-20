// Payload encoding and decoding for the versioned native model format.
//
// The payload contains only quantization schema, objective metadata, and flat
// trees. Container framing, checksums, filesystem paths, and devices stay out.

#include "model_format_internal.hpp"

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "mpsboost/objective.hpp"

namespace mpsboost::model_format_internal {

std::vector<std::uint8_t> BuildPayload(const RegressionModel& model) {
  std::vector<std::uint8_t> payload;
  AppendUnsigned(&payload, model.feature_count());
  AppendUnsigned(&payload, model.schema().max_bins());
  AppendFloat<double, std::uint64_t>(&payload, model.base_score());
  AppendFloat<double, std::uint64_t>(&payload, model.learning_rate());
  AppendUnsigned(
      &payload,
      static_cast<std::uint32_t>(
          model.objective() == TrainingParameters::Objective::kBinaryLogistic
              ? 1
              : 0));
  AppendUnsigned(
      &payload,
      static_cast<std::uint64_t>(model.schema().boundaries().size()));
  for (const FeatureBinMetadata& item : model.schema().feature_metadata()) {
    AppendUnsigned(&payload, item.boundary_offset);
    AppendUnsigned(&payload, item.boundary_count);
    AppendUnsigned(&payload, item.bin_count);
    AppendUnsigned(&payload, item.missing_count);
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
      AppendUnsigned(&payload, static_cast<std::uint8_t>(node.default_left ? 1 : 0));
      AppendUnsigned(&payload, node.flags);
      for (int index = 0; index < 6; ++index) {
        AppendUnsigned(&payload, std::uint8_t{0});
      }
    }
  }
  return payload;
}

RegressionModel ParsePayload(const std::uint8_t* data,
                             std::size_t size,
                             std::uint16_t format_minor) {
  Reader reader(data, size);
  const std::uint32_t features =
      reader.ReadUnsigned<std::uint32_t>("features");
  const std::uint32_t max_bins =
      reader.ReadUnsigned<std::uint32_t>("max_bins");
  const double base_score =
      reader.ReadFloat<double, std::uint64_t>("base_score");
  const double learning_rate =
      reader.ReadFloat<double, std::uint64_t>("learning_rate");
  TrainingParameters::Objective objective =
      TrainingParameters::Objective::kSquaredError;
  if (format_minor >= 1) {
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
  if (boundary_count > size / sizeof(float)) {
    throw TrainingError("模型 boundary_count 超出 payload 范围");
  }
  std::vector<FeatureBinMetadata> metadata;
  metadata.reserve(features);
  for (std::uint32_t feature = 0; feature < features; ++feature) {
    FeatureBinMetadata item{
        reader.ReadUnsigned<std::uint64_t>("boundary_offset"),
        reader.ReadUnsigned<std::uint32_t>("feature boundary_count"),
        reader.ReadUnsigned<std::uint32_t>("feature bin_count"),
        0};
    if (format_minor >= 2) {
      item.missing_count = reader.ReadUnsigned<std::uint64_t>("feature missing_count");
    }
    metadata.push_back(item);
  }
  std::vector<float> boundaries;
  boundaries.reserve(static_cast<std::size_t>(boundary_count));
  for (std::uint64_t index = 0; index < boundary_count; ++index) {
    boundaries.push_back(reader.ReadFloat<float, std::uint32_t>("boundary"));
  }
  QuantizationSchema schema = RestoreQuantizationSchema(
      features, max_bins, std::move(boundaries), std::move(metadata));

  const std::uint32_t tree_count =
      reader.ReadUnsigned<std::uint32_t>("tree_count");
  if (tree_count == 0 || tree_count > size / 41) {
    throw TrainingError("模型树数量不合法");
  }
  std::vector<RegressionTree> trees;
  trees.reserve(tree_count);
  for (std::uint32_t tree = 0; tree < tree_count; ++tree) {
    const std::uint32_t node_count =
        reader.ReadUnsigned<std::uint32_t>("node_count");
    if (node_count == 0 || node_count > size / 41) {
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
      if (format_minor >= 2) {
        const std::uint8_t default_left =
            reader.ReadUnsigned<std::uint8_t>("default_left");
        if (default_left > 1) {
          throw TrainingError("model default_left must be 0 or 1");
        }
        node.default_left = default_left != 0;
      }
      node.flags = reader.ReadUnsigned<std::uint8_t>("flags");
      const int reserved_count = format_minor >= 2 ? 6 : 7;
      for (int reserved = 0; reserved < reserved_count; ++reserved) {
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
  return RegressionModel::Restore(std::move(schema), base_score, learning_rate,
                                  objective, std::move(trees));
}

}  // namespace mpsboost::model_format_internal

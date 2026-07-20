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
namespace {

void AppendSchema(std::vector<std::uint8_t>* payload,
                  const QuantizationSchema& schema) {
  AppendUnsigned(payload, schema.features());
  AppendUnsigned(payload, schema.max_bins());
  AppendUnsigned(payload, static_cast<std::uint64_t>(schema.boundaries().size()));
  for (const FeatureBinMetadata& item : schema.feature_metadata()) {
    AppendUnsigned(payload, item.boundary_offset);
    AppendUnsigned(payload, item.boundary_count);
    AppendUnsigned(payload, item.bin_count);
    AppendUnsigned(payload, item.missing_count);
  }
  for (const float boundary : schema.boundaries()) {
    AppendFloat<float, std::uint32_t>(payload, boundary);
  }
}

QuantizationSchema ParseSchema(Reader* reader, std::size_t size,
                               std::uint16_t format_minor) {
  const std::uint32_t features =
      reader->ReadUnsigned<std::uint32_t>("features");
  const std::uint32_t max_bins =
      reader->ReadUnsigned<std::uint32_t>("max_bins");
  const std::uint64_t boundary_count =
      reader->ReadUnsigned<std::uint64_t>("boundary_count");
  if (boundary_count > size / sizeof(float)) {
    throw TrainingError("模型 boundary_count 超出 payload 范围");
  }
  std::vector<FeatureBinMetadata> metadata;
  metadata.reserve(features);
  for (std::uint32_t feature = 0; feature < features; ++feature) {
    FeatureBinMetadata item{
        reader->ReadUnsigned<std::uint64_t>("boundary_offset"),
        reader->ReadUnsigned<std::uint32_t>("feature boundary_count"),
        reader->ReadUnsigned<std::uint32_t>("feature bin_count"),
        0};
    if (format_minor >= 2) {
      item.missing_count =
          reader->ReadUnsigned<std::uint64_t>("feature missing_count");
    }
    metadata.push_back(item);
  }
  std::vector<float> boundaries;
  boundaries.reserve(static_cast<std::size_t>(boundary_count));
  for (std::uint64_t index = 0; index < boundary_count; ++index) {
    boundaries.push_back(reader->ReadFloat<float, std::uint32_t>("boundary"));
  }
  return RestoreQuantizationSchema(features, max_bins, std::move(boundaries),
                                   std::move(metadata));
}

void AppendTrees(std::vector<std::uint8_t>* payload,
                 const std::vector<RegressionTree>& trees) {
  AppendUnsigned(payload, static_cast<std::uint32_t>(trees.size()));
  for (const RegressionTree& tree : trees) {
    if (tree.nodes().size() > std::numeric_limits<std::uint32_t>::max()) {
      throw TrainingError("模型树节点数超出格式限制");
    }
    AppendUnsigned(payload, static_cast<std::uint32_t>(tree.nodes().size()));
    for (const TreeNode& node : tree.nodes()) {
      AppendUnsigned(payload, node.feature_index);
      AppendUnsigned(payload, node.threshold_bin);
      AppendUnsigned(payload, node.left_child);
      AppendUnsigned(payload, node.right_child);
      AppendFloat<double, std::uint64_t>(payload, node.leaf_value);
      AppendFloat<double, std::uint64_t>(payload, node.gain);
      AppendUnsigned(payload, static_cast<std::uint8_t>(node.default_left ? 1 : 0));
      AppendUnsigned(payload, node.flags);
      for (int index = 0; index < 6; ++index) {
        AppendUnsigned(payload, std::uint8_t{0});
      }
    }
  }
}

std::vector<RegressionTree> ParseTrees(Reader* reader, std::size_t size,
                                       std::uint16_t format_minor,
                                       std::uint32_t feature_count) {
  const std::uint32_t tree_count =
      reader->ReadUnsigned<std::uint32_t>("tree_count");
  if (tree_count == 0 || tree_count > size / 41) {
    throw TrainingError("模型树数量不合法");
  }
  std::vector<RegressionTree> trees;
  trees.reserve(tree_count);
  for (std::uint32_t tree = 0; tree < tree_count; ++tree) {
    const std::uint32_t node_count =
        reader->ReadUnsigned<std::uint32_t>("node_count");
    if (node_count == 0 || node_count > size / 41) {
      throw TrainingError("模型节点数量不合法");
    }
    std::vector<TreeNode> nodes;
    nodes.reserve(node_count);
    for (std::uint32_t node_index = 0; node_index < node_count; ++node_index) {
      TreeNode node;
      node.feature_index = reader->ReadUnsigned<std::uint32_t>("feature_index");
      node.threshold_bin = reader->ReadUnsigned<std::uint32_t>("threshold_bin");
      node.left_child = reader->ReadUnsigned<std::uint32_t>("left_child");
      node.right_child = reader->ReadUnsigned<std::uint32_t>("right_child");
      node.leaf_value =
          reader->ReadFloat<double, std::uint64_t>("leaf_value");
      node.gain = reader->ReadFloat<double, std::uint64_t>("gain");
      if (format_minor >= 2) {
        const std::uint8_t default_left =
            reader->ReadUnsigned<std::uint8_t>("default_left");
        if (default_left > 1) {
          throw TrainingError("model default_left must be 0 or 1");
        }
        node.default_left = default_left != 0;
      }
      node.flags = reader->ReadUnsigned<std::uint8_t>("flags");
      const int reserved_count = format_minor >= 2 ? 6 : 7;
      for (int reserved = 0; reserved < reserved_count; ++reserved) {
        if (reader->ReadUnsigned<std::uint8_t>("node reserved") != 0) {
          throw TrainingError("模型节点保留字段必须为零");
        }
      }
      nodes.push_back(node);
    }
    trees.push_back(RegressionTree::Restore(feature_count, std::move(nodes)));
  }
  return trees;
}

}  // namespace

std::vector<std::uint8_t> BuildPayload(const RegressionModel& model) {
  std::vector<std::uint8_t> payload;
  AppendUnsigned(&payload, model.feature_count());
  AppendUnsigned(&payload, model.schema().max_bins());
  AppendFloat<double, std::uint64_t>(&payload, model.base_score());
  AppendFloat<double, std::uint64_t>(&payload, model.learning_rate());
  AppendUnsigned(
      &payload,
      static_cast<std::uint32_t>(model.objective()));
  AppendFloat<double, std::uint64_t>(&payload, model.objective_alpha());
  AppendFloat<double, std::uint64_t>(&payload,
                                     model.tweedie_variance_power());
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
  AppendTrees(&payload, model.trees());
  return payload;
}

std::vector<std::uint8_t> BuildPayload(const MulticlassModel& model) {
  std::vector<std::uint8_t> payload;
  AppendUnsigned(&payload, model.class_count());
  AppendFloat<double, std::uint64_t>(&payload, model.learning_rate());
  for (const double base_score : model.base_scores()) {
    AppendFloat<double, std::uint64_t>(&payload, base_score);
  }
  for (const double class_label : model.class_labels()) {
    AppendFloat<double, std::uint64_t>(&payload, class_label);
  }
  AppendSchema(&payload, model.schema());
  AppendTrees(&payload, model.trees());
  return payload;
}

RegressionModel ParsePayload(const std::uint8_t* data,
                             std::size_t size,
                             std::uint16_t format_minor) {
  Reader reader(data, size);
  const std::uint32_t features = reader.ReadUnsigned<std::uint32_t>("features");
  const std::uint32_t max_bins = reader.ReadUnsigned<std::uint32_t>("max_bins");
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
    } else if (objective_code == 2) {
      objective = TrainingParameters::Objective::kQuantile;
    } else if (objective_code == 3) {
      objective = TrainingParameters::Objective::kPoisson;
    } else if (objective_code == 4) {
      objective = TrainingParameters::Objective::kTweedie;
    } else {
      throw TrainingError("model objective is unsupported");
    }
  }
  double objective_alpha = 0.5;
  double tweedie_variance_power = 1.5;
  if (format_minor >= 4) {
    objective_alpha =
        reader.ReadFloat<double, std::uint64_t>("objective_alpha");
    tweedie_variance_power =
        reader.ReadFloat<double, std::uint64_t>("tweedie_variance_power");
  }
  const std::uint64_t boundary_count = reader.ReadUnsigned<std::uint64_t>("boundary_count");
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

  std::vector<RegressionTree> trees =
      ParseTrees(&reader, size, format_minor, features);
  if (!reader.at_end()) {
    throw TrainingError("模型包含未识别的尾部字节");
  }
  return RegressionModel::Restore(std::move(schema), base_score, learning_rate,
                                  objective, objective_alpha,
                                  tweedie_variance_power, std::move(trees));
}

MulticlassModel ParseMulticlassPayload(const std::uint8_t* data,
                                       std::size_t size,
                                       std::uint16_t format_minor) {
  if (format_minor < 3) {
    throw TrainingError("multiclass model format is unsupported");
  }
  Reader reader(data, size);
  const std::uint32_t class_count =
      reader.ReadUnsigned<std::uint32_t>("class_count");
  const double learning_rate =
      reader.ReadFloat<double, std::uint64_t>("learning_rate");
  std::vector<double> base_scores;
  base_scores.reserve(class_count);
  for (std::uint32_t class_index = 0; class_index < class_count; ++class_index) {
    base_scores.push_back(
        reader.ReadFloat<double, std::uint64_t>("softmax base_score"));
  }
  std::vector<double> class_labels;
  class_labels.reserve(class_count);
  for (std::uint32_t class_index = 0; class_index < class_count; ++class_index) {
    class_labels.push_back(
        reader.ReadFloat<double, std::uint64_t>("softmax class label"));
  }
  QuantizationSchema schema = ParseSchema(&reader, size, format_minor);
  std::vector<RegressionTree> trees =
      ParseTrees(&reader, size, format_minor, schema.features());
  if (!reader.at_end()) {
    throw TrainingError("模型包含未识别的尾部字节");
  }
  return MulticlassModel::Restore(std::move(schema), class_count, learning_rate,
                                  std::move(base_scores),
                                  std::move(class_labels), std::move(trees));
}

}  // namespace mpsboost::model_format_internal

// MPSBoost GPU row partition and compaction kernel.
//
// Responsibility: stably partitions node row indices into left and right segments
// using the feature/threshold confirmed by the training core. The kernel neither
// selects splits nor creates nodes; it only performs verifiable data movement. It
// currently writes sequentially from one GPU thread to preserve deterministic order.
// Any future parallel prefix-sum implementation must preserve that order.

#include <metal_stdlib>

using namespace metal;

template <typename BinType>
inline void PartitionRows(
    device const BinType* bins,
    device const uint* input_rows,
    device uint* left_rows,
    device uint* right_rows,
    device atomic_uint* counts,
    constant uint& dataset_rows,
    constant uint& selected_rows,
    constant uint& feature,
    constant uint& threshold_bin,
    uint index) {
  if (index != 0) {
    return;
  }
  uint left_count = 0;
  uint right_count = 0;
  for (uint position = 0; position < selected_rows; ++position) {
    const uint row = input_rows[position];
    if (row >= dataset_rows) {
      continue;
    }
    if (uint(bins[feature * dataset_rows + row]) <= threshold_bin) {
      left_rows[left_count++] = row;
    } else {
      right_rows[right_count++] = row;
    }
  }
  atomic_store_explicit(&counts[0], left_count, memory_order_relaxed);
  atomic_store_explicit(&counts[1], right_count, memory_order_relaxed);
}

#define PARTITION_KERNEL(NAME, BIN_TYPE)                                         \
  kernel void NAME(                                                              \
      device const BIN_TYPE* bins [[buffer(0)]],                                 \
      device const uint* input_rows [[buffer(1)]],                               \
      device uint* left_rows [[buffer(2)]],                                      \
      device uint* right_rows [[buffer(3)]],                                     \
      device atomic_uint* counts [[buffer(4)]],                                  \
      constant uint& dataset_rows [[buffer(5)]],                                 \
      constant uint& selected_rows [[buffer(6)]],                                \
      constant uint& feature [[buffer(7)]],                                      \
      constant uint& threshold_bin [[buffer(8)]],                                \
      uint index [[thread_position_in_grid]]) {                                  \
    PartitionRows(bins, input_rows, left_rows, right_rows, counts, dataset_rows, \
                  selected_rows, feature, threshold_bin, index);                 \
  }

PARTITION_KERNEL(partition_rows_u8, uchar)
PARTITION_KERNEL(partition_rows_u16, ushort)

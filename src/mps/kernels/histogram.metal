// MPSBoost 两阶段分层直方图 kernel。
//
// 第一阶段让每个 threadgroup 负责一个 feature/bin/partial，在线程组内归约 count/G/H；
// 第二阶段合并有限数量的 partial。partial 数有硬上限，避免工作区随 rows×features×bins
// 无界增长。本文件不选择 split，也不定义树生长语义。

#include <metal_stdlib>

using namespace metal;

struct HistogramValue {
  uint count;
  float gradient;
  float hessian;
  uint reserved;
};

// 公共实现显式接收 cell_count 作为 partial stride；保留单一模板主体，两个存储宽度
// wrapper 只负责类型实例化，避免 uint8/uint16 演化出两套统计语义。
inline void AtomicAddFloat(threadgroup atomic_uint* address, float operand) {
  uint expected = atomic_load_explicit(address, memory_order_relaxed);
  while (true) {
    const uint desired = as_type<uint>(as_type<float>(expected) + operand);
    if (atomic_compare_exchange_weak_explicit(
            address, &expected, desired, memory_order_relaxed,
            memory_order_relaxed)) {
      return;
    }
  }
}

template <typename BinType>
inline void BuildFeaturePartial(
    device const BinType* bins,
    device const uint* rows,
    device const float2* gradients,
    device HistogramValue* partials,
    device const uint* feature_offsets,
    device const uint* feature_bin_counts,
    constant uint& dataset_rows,
    constant uint& selected_rows,
    constant uint& partial_count,
    constant uint& cell_count,
    uint3 group_position,
    uint thread_index,
    uint thread_count,
    threadgroup atomic_uint* local_count,
    threadgroup atomic_uint* local_gradient,
    threadgroup atomic_uint* local_hessian) {
  const uint feature = group_position.x;
  const uint bin_count = feature_bin_counts[feature];
  for (uint bin = thread_index; bin < bin_count; bin += thread_count) {
    atomic_store_explicit(&local_count[bin], 0, memory_order_relaxed);
    atomic_store_explicit(&local_gradient[bin], as_type<uint>(0.0F),
                          memory_order_relaxed);
    atomic_store_explicit(&local_hessian[bin], as_type<uint>(0.0F),
                          memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 每个 feature/partial 只扫描自己分到的行，复杂度保持 O(rows×features)。原子冲突
  // 被限制在线程组局部内存，跨 threadgroup 不争用同一个全局 bin。
  for (uint position = group_position.y * thread_count + thread_index;
       position < selected_rows;
       position += partial_count * thread_count) {
    const uint row = rows[position];
    if (row < dataset_rows) {
      const uint bin = uint(bins[feature * dataset_rows + row]);
      if (bin < bin_count) {
        atomic_fetch_add_explicit(&local_count[bin], 1, memory_order_relaxed);
        AtomicAddFloat(&local_gradient[bin], gradients[row].x);
        AtomicAddFloat(&local_hessian[bin], gradients[row].y);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint feature_offset = feature_offsets[feature];
  for (uint bin = thread_index; bin < bin_count; bin += thread_count) {
    partials[group_position.y * cell_count + feature_offset + bin] = HistogramValue{
        atomic_load_explicit(&local_count[bin], memory_order_relaxed),
        as_type<float>(atomic_load_explicit(&local_gradient[bin], memory_order_relaxed)),
        as_type<float>(atomic_load_explicit(&local_hessian[bin], memory_order_relaxed)),
        0};
  }
}

#define HISTOGRAM_PARTIAL_KERNEL(NAME, BIN_TYPE)                                      \
kernel void NAME(                                                                     \
    device const BIN_TYPE* bins [[buffer(0)]], device const uint* rows [[buffer(1)]], \
    device const float2* gradients [[buffer(2)]],                                     \
    device HistogramValue* partials [[buffer(3)]],                                    \
    device const uint* feature_offsets [[buffer(4)]],                                 \
    device const uint* feature_bin_counts [[buffer(5)]],                              \
    constant uint& dataset_rows [[buffer(6)]],                                        \
    constant uint& selected_rows [[buffer(7)]],                                      \
    constant uint& partial_count [[buffer(8)]], constant uint& cell_count [[buffer(9)]],\
    uint3 group_position [[threadgroup_position_in_grid]],                            \
    uint3 thread_position [[thread_position_in_threadgroup]],                         \
    uint3 thread_count [[threads_per_threadgroup]],                                   \
    threadgroup atomic_uint* local_count [[threadgroup(0)]],                          \
    threadgroup atomic_uint* local_gradient [[threadgroup(1)]],                       \
    threadgroup atomic_uint* local_hessian [[threadgroup(2)]]) {                      \
  BuildFeaturePartial(bins, rows, gradients, partials, feature_offsets,               \
                      feature_bin_counts, dataset_rows, selected_rows, partial_count, \
                      cell_count, group_position, thread_position.x, thread_count.x,  \
                      local_count, local_gradient, local_hessian);                    \
}

HISTOGRAM_PARTIAL_KERNEL(histogram_partial_u8, uchar)
HISTOGRAM_PARTIAL_KERNEL(histogram_partial_u16, ushort)

template <typename BinType>
inline void BuildBaseline(
    device const BinType* bins,
    device const uint* rows,
    device const float2* gradients,
    device HistogramValue* output,
    device const uint* cell_features,
    device const uint* cell_bins,
    constant uint& dataset_rows,
    constant uint& selected_rows,
    constant uint& cell_count,
    uint cell) {
  if (cell >= cell_count) {
    return;
  }
  const uint feature = cell_features[cell];
  const uint target_bin = cell_bins[cell];
  HistogramValue total{0, 0.0F, 0.0F, 0};
  // 一线程顺序扫描一个 cell，作为最简单的 GPU 正确性基线。该路径不用于生产训练，
  // 但能把分层归约错误与 host/数据布局错误分离定位。
  for (uint position = 0; position < selected_rows; ++position) {
    const uint row = rows[position];
    if (row < dataset_rows && uint(bins[feature * dataset_rows + row]) == target_bin) {
      ++total.count;
      total.gradient += gradients[row].x;
      total.hessian += gradients[row].y;
    }
  }
  output[cell] = total;
}

#define HISTOGRAM_BASELINE_KERNEL(NAME, BIN_TYPE)                                     \
kernel void NAME(                                                                     \
    device const BIN_TYPE* bins [[buffer(0)]], device const uint* rows [[buffer(1)]], \
    device const float2* gradients [[buffer(2)]],                                     \
    device HistogramValue* output [[buffer(3)]],                                      \
    device const uint* cell_features [[buffer(4)]],                                   \
    device const uint* cell_bins [[buffer(5)]],                                       \
    constant uint& dataset_rows [[buffer(6)]],                                        \
    constant uint& selected_rows [[buffer(7)]], constant uint& cell_count [[buffer(8)]],\
    uint cell [[thread_position_in_grid]]) {                                           \
  BuildBaseline(bins, rows, gradients, output, cell_features, cell_bins,              \
                dataset_rows, selected_rows, cell_count, cell);                       \
}

HISTOGRAM_BASELINE_KERNEL(histogram_baseline_u8, uchar)
HISTOGRAM_BASELINE_KERNEL(histogram_baseline_u16, ushort)

kernel void histogram_reduce(
    device const HistogramValue* partials [[buffer(0)]],
    device HistogramValue* output [[buffer(1)]],
    constant uint& cell_count [[buffer(2)]],
    constant uint& partial_count [[buffer(3)]],
    uint cell [[thread_position_in_grid]]) {
  if (cell >= cell_count) {
    return;
  }
  HistogramValue total{0, 0.0F, 0.0F, 0};
  for (uint partial = 0; partial < partial_count; ++partial) {
    const HistogramValue value = partials[partial * cell_count + cell];
    total.count += value.count;
    total.gradient += value.gradient;
    total.hessian += value.hessian;
  }
  output[cell] = total;
}

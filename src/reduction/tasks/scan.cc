/* Copyright 2021 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "reduction/tasks/scan.h"
#include "reduction/reduction_op.h"

namespace legate {
namespace pandas {
namespace reduction {

using namespace Legion;

using ScanArgs = ScanTask::ScanArgs;

void ScanTask::ScanArgs::sanity_check(void)
{
  if (local) assert(input.num_elements() == output.num_elements());
}

namespace detail {

template <typename T>
using Scalar = std::pair<T, bool>;

template <typename T, typename Op>
Scalar<T> scan(T *output,
               Bitmask *maybe_output_b,
               const T *input,
               Bitmask *maybe_input_b,
               size_t size,
               bool skipna)
{
  Op op{};
  Scalar<T> agg{Op::identity(), false};

  auto apply_op = [&](auto &out, auto &in) {
    auto local_agg = op(in, agg.first);
    out            = local_agg;
    agg.first      = local_agg;
    agg.second     = true;
  };

  if (nullptr == maybe_input_b)
    for (auto idx = 0; idx < size; ++idx) apply_op(output[idx], input[idx]);
  else {
    auto &input_b = *maybe_input_b;
#ifdef DEBUG_PANDAS
    assert(nullptr != maybe_output_b);
#endif
    auto &output_b = *maybe_output_b;
    if (skipna) {
      input_b.copy(output_b);
      for (auto idx = 0; idx < size; ++idx) {
        if (!input_b.get(idx)) continue;
        apply_op(output[idx], input[idx]);
      }
    } else {
      output_b.clear();
      for (auto idx = 0; idx < size; ++idx) {
        if (!input_b.get(idx)) {
          agg.second = false;
          break;
        }
        apply_op(output[idx], input[idx]);
        output_b.set(idx);
      }
    }
  }

  return agg;
}

template <TypeCode CODE>
void dispatch_scan(ScanArgs &args)
{
  using T = pandas_type_of<CODE>;

  auto input    = args.input.raw_column_read<T>();
  auto input_b  = args.input.maybe_read_bitmask();
  auto output   = args.output.raw_column_write<T>();
  auto output_b = args.output.maybe_write_bitmask();
  auto size     = args.input.num_elements();

  Scalar<T> local_agg;
  switch (args.code) {
    case AggregationCode::SUM: {
      local_agg = scan<T, Op<AggregationCode::SUM, T>>(
        output, output_b.get(), input, input_b.get(), size, args.skipna);
      break;
    }
    case AggregationCode::MAX: {
      local_agg = scan<T, Op<AggregationCode::MAX, T>>(
        output, output_b.get(), input, input_b.get(), size, args.skipna);
      break;
    }
    case AggregationCode::MIN: {
      local_agg = scan<T, Op<AggregationCode::MIN, T>>(
        output, output_b.get(), input, input_b.get(), size, args.skipna);
      break;
    }
    case AggregationCode::PROD: {
      local_agg = scan<T, Op<AggregationCode::PROD, T>>(
        output, output_b.get(), input, input_b.get(), size, args.skipna);
      break;
    }
    default: {
      assert(false);
      break;
    }
  }

  if (args.has_buffer) {
    auto buffer = args.write_buffer.raw_column_write<T>();
    buffer[0]   = local_agg.first;
    if (args.input.nullable()) {
      auto buffer_b = args.write_buffer.write_bitmask();
      buffer_b.set(0, local_agg.second);
    }
  }
}

template <typename T, typename Op>
void apply(T *output,
           Bitmask *maybe_output_b,
           size_t size,
           T *buffer,
           Bitmask *maybe_buffer_b,
           coord_t idx,
           bool skipna)
{
  T value = buffer[idx - 1];

  Op op{};
  if (nullptr == maybe_output_b)
    for (auto idx = 0; idx < size; ++idx) output[idx] = op(output[idx], value);
  else {
    if (skipna) {
      bool valid = maybe_buffer_b->get(idx - 1);
      if (!valid) return;
      for (auto idx = 0; idx < size; ++idx) output[idx] = op(output[idx], value);
    } else {
      auto &output_b = *maybe_output_b;
      bool valid     = maybe_buffer_b->get(idx - 1);
      if (!valid)
        output_b.clear();
      else
        for (auto idx = 0; idx < size; ++idx) {
          if (!output_b.get(idx)) break;
          output[idx] = op(output[idx], value);
        }
    }
  }
}

template <TypeCode CODE>
void dispatch_apply(ScanArgs &args, coord_t idx)
{
  using T = pandas_type_of<CODE>;

  if (idx == 0) return;

  auto buffer      = args.read_buffer.raw_column_read<T>();
  auto buffer_b    = args.read_buffer.maybe_read_bitmask();
  auto buffer_size = args.read_buffer.num_elements();

  T global_agg[buffer_size];
  Bitmask::AllocType raw_global_agg_b[buffer_size];
  Bitmask global_agg_b(raw_global_agg_b, buffer_size);
  auto maybe_global_agg_b = nullptr != buffer_b ? &global_agg_b : nullptr;

  switch (args.code) {
    case AggregationCode::SUM: {
      scan<T, Op<AggregationCode::SUM, T>>(
        global_agg, maybe_global_agg_b, buffer, buffer_b.get(), buffer_size, args.skipna);
      break;
    }
    case AggregationCode::MAX: {
      scan<T, Op<AggregationCode::MAX, T>>(
        global_agg, maybe_global_agg_b, buffer, buffer_b.get(), buffer_size, args.skipna);
      break;
    }
    case AggregationCode::MIN: {
      scan<T, Op<AggregationCode::MIN, T>>(
        global_agg, maybe_global_agg_b, buffer, buffer_b.get(), buffer_size, args.skipna);
      break;
    }
    case AggregationCode::PROD: {
      scan<T, Op<AggregationCode::PROD, T>>(
        global_agg, maybe_global_agg_b, buffer, buffer_b.get(), buffer_size, args.skipna);
      break;
    }
    default: {
      assert(false);
      break;
    }
  }

  auto output   = args.output.raw_column_write<T>();
  auto output_b = args.output.maybe_write_bitmask();
  auto size     = args.output.num_elements();

  switch (args.code) {
    case AggregationCode::SUM: {
      apply<T, Op<AggregationCode::SUM, T>>(
        output, output_b.get(), size, global_agg, maybe_global_agg_b, idx, args.skipna);
      break;
    }
    case AggregationCode::MAX: {
      apply<T, Op<AggregationCode::MAX, T>>(
        output, output_b.get(), size, global_agg, maybe_global_agg_b, idx, args.skipna);
      break;
    }
    case AggregationCode::MIN: {
      apply<T, Op<AggregationCode::MIN, T>>(
        output, output_b.get(), size, global_agg, maybe_global_agg_b, idx, args.skipna);
      break;
    }
    case AggregationCode::PROD: {
      apply<T, Op<AggregationCode::PROD, T>>(
        output, output_b.get(), size, global_agg, maybe_global_agg_b, idx, args.skipna);
      break;
    }
    default: {
      assert(false);
      break;
    }
  }
}

struct ScanOrApply {
  template <TypeCode CODE, std::enable_if_t<is_numeric_type<CODE>::value> * = nullptr>
  void operator()(ScanArgs &args, int64_t task_id)
  {
    if (args.local)
      detail::dispatch_scan<CODE>(args);
    else
      detail::dispatch_apply<CODE>(args, task_id);
  }

  template <TypeCode CODE, std::enable_if_t<!is_numeric_type<CODE>::value> * = nullptr>
  void operator()(ScanArgs &args, int64_t task_id)
  {
    assert(false);
  }
};

}  // namespace detail

/*static*/ void ScanTask::cpu_variant(const Task *task,
                                      const std::vector<PhysicalRegion> &regions,
                                      Context context,
                                      Runtime *runtime)
{
  Deserializer ctx{task, regions};

  ScanArgs args;
  deserialize(ctx, args);

  if (args.output.empty()) return;

  type_dispatch(args.output.code(), detail::ScanOrApply{}, args, task->index_point[0]);
}

void deserialize(Deserializer &ctx, ScanTask::ScanArgs &args)
{
  deserialize(ctx, args.local);
  deserialize(ctx, args.code);
  deserialize(ctx, args.skipna);
  if (args.local) {
    deserialize(ctx, args.output);
    deserialize(ctx, args.input);
    deserialize(ctx, args.has_buffer);
    if (args.has_buffer) deserialize(ctx, args.write_buffer);
  } else {
    deserialize(ctx, args.output);
    deserialize(ctx, args.read_buffer);
  }
#ifdef DEBUG_PANDAS
  args.sanity_check();
#endif
}

static void __attribute__((constructor)) register_tasks(void) { ScanTask::register_variants(); }

}  // namespace reduction
}  // namespace pandas
}  // namespace legate

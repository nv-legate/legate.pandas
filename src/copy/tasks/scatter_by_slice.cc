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

#include "copy/tasks/scatter_by_slice.h"
#include "copy/copy.h"
#include "column/detail/column.h"
#include "util/allocator.h"
#include "util/type_dispatch.h"

namespace legate {
namespace pandas {
namespace copy {

using namespace Legion;

using ScatterBySliceArg = ScatterBySliceTask::ScatterBySliceTaskArgs::ScatterBySliceArg;

void ScatterBySliceTask::ScatterBySliceTaskArgs::sanity_check(void)
{
  auto target_range  = range.intersection(requests.front().target.shape());
  auto target_volume = target_range.volume();

  for (auto &req : requests) {
    if (!input_is_scalar) assert(req.input.code() == req.target.code());
    assert(req.target.shape() == req.output.shape());
    if (!input_is_scalar) assert(req.input.num_elements() == target_volume);
  }
}

using ColumnView = pandas::detail::Column;

namespace detail {

struct UpdateRange {
  coord_t start{0};
  coord_t stop{0};
};

UpdateRange compute_update_range(const Rect<1> &target_bounds, const Rect<1> &target_range)
{
  UpdateRange to_update{};
  if (!target_bounds.intersection(target_range).empty()) {
    auto target_range_start =
      std::min(target_bounds.hi[0], std::max(target_bounds.lo[0], target_range.lo[0]));
    auto target_range_stop =
      std::max(target_bounds.lo[0], std::min(target_bounds.hi[0], target_range.hi[0]));

    to_update.start = target_range_start - target_bounds.lo[0];
    to_update.stop  = target_range_stop - target_bounds.lo[0] + 1;
  }

  return to_update;
}

template <typename TargetAccessor, typename InputAccessor>
struct range_scatter_fn
  : public thrust::unary_function<coord_t, typename TargetAccessor::value_type> {
  range_scatter_fn(const UpdateRange &to_update, TargetAccessor &&target, InputAccessor &&input)
    : to_update_(to_update), target_(target), input_(input)
  {
  }

  typename TargetAccessor::value_type operator()(coord_t idx)
  {
    if (to_update_.start <= idx && idx < to_update_.stop)
      return input_(idx - to_update_.start);
    else
      return target_(idx);
  }

  UpdateRange to_update_;
  TargetAccessor target_;
  InputAccessor input_;
};

template <typename TargetAccessor, typename InputAccessor>
decltype(auto) make_range_scatter_fn(const UpdateRange &to_update,
                                     TargetAccessor &&target,
                                     InputAccessor &&input)
{
  return range_scatter_fn<TargetAccessor, InputAccessor>(
    to_update, std::forward<TargetAccessor>(target), std::forward<InputAccessor>(input));
}

template <typename OutputIterator, typename TargetAccessor, typename InputAccessor>
void copy_range(OutputIterator &&output,
                UpdateRange &to_update,
                TargetAccessor &&target,
                InputAccessor &&input,
                size_t size)
{
  auto fn = make_range_scatter_fn(
    to_update, std::forward<TargetAccessor>(target), std::forward<InputAccessor>(input));
  auto start = thrust::make_counting_iterator<coord_t>(0);
  auto stop  = thrust::make_counting_iterator<coord_t>(size);
  thrust::transform(thrust::host, start, stop, output, fn);
}

template <typename TargetAccessor, typename InputAccessor>
int32_t sum(UpdateRange &to_update, TargetAccessor &&target, InputAccessor &&input, size_t size)
{
  auto fn = make_range_scatter_fn(
    to_update, std::forward<TargetAccessor>(target), std::forward<InputAccessor>(input));
  auto start = thrust::make_transform_iterator(thrust::make_counting_iterator<coord_t>(0), fn);
  auto stop  = thrust::make_transform_iterator(thrust::make_counting_iterator<coord_t>(size), fn);
  return thrust::reduce(thrust::host, start, stop);
}

struct CopyRange {
  template <TypeCode CODE, std::enable_if_t<is_primitive_type<CODE>::value> * = nullptr>
  ColumnView operator()(const Rect<1> &target_bounds,
                        const Rect<1> &target_range,
                        const ColumnView &target,
                        const ColumnView &input,
                        alloc::Allocator &allocator)
  {
    using VAL = pandas_type_of<CODE>;

    auto size  = target.size();
    auto p_out = allocator.allocate_elements<VAL>(size);

    auto to_update = compute_update_range(target_bounds, target_range);

    copy_range(p_out, to_update, accessor_fn<VAL>(target), accessor_fn<VAL>(input), size);

    auto p_out_b = static_cast<Bitmask::AllocType *>(nullptr);
    if (target.nullable()) {
      p_out_b = allocator.allocate_elements<Bitmask::AllocType>(size);
      if (input.nullable())
        copy_range(p_out_b,
                   to_update,
                   bitmask_accessor_fn(target.bitmask()),
                   bitmask_accessor_fn(input.bitmask()),
                   size);
      else
        copy_range(p_out_b,
                   to_update,
                   bitmask_accessor_fn(target.bitmask()),
                   broadcast_fn<Bitmask::AllocType>(true),
                   size);
    }

    return ColumnView(CODE, p_out, size, p_out_b);
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::STRING> * = nullptr>
  ColumnView operator()(const Rect<1> &target_bounds,
                        const Rect<1> &target_range,
                        const ColumnView &target,
                        const ColumnView &input,
                        alloc::Allocator &allocator)
  {
    auto size = target.size();

    auto to_update = compute_update_range(target_bounds, target_range);

    auto num_chars =
      sum(to_update, size_accessor_fn(target.child(0)), size_accessor_fn(input.child(0)), size);

    auto p_out_o = allocator.allocate_elements<int32_t>(size + 1);
    auto p_out_c = allocator.allocate_elements<int8_t>(num_chars);

    copy_range(p_out_o,
               to_update,
               size_accessor_fn(target.child(0)),
               size_accessor_fn(input.child(0)),
               size);
    p_out_o[size] = 0;
    thrust::exclusive_scan(thrust::host, p_out_o, p_out_o + size + 1, p_out_o);

    auto fn = make_range_scatter_fn(
      to_update, accessor_fn<std::string>(target), accessor_fn<std::string>(input));

    auto ptr = p_out_c;
    for (coord_t idx = 0; idx < size; ++idx) {
      std::string value = fn(idx);
      memcpy(ptr, value.c_str(), value.size());
      ptr += value.size();
    }
#ifdef DEBUG_PANDAS
    assert(ptr - p_out_c == num_chars);
#endif

    auto p_out_b = static_cast<Bitmask::AllocType *>(nullptr);
    if (target.nullable()) {
      p_out_b = allocator.allocate_elements<Bitmask::AllocType>(size);
      if (input.nullable())
        copy_range(p_out_b,
                   to_update,
                   bitmask_accessor_fn(target.bitmask()),
                   bitmask_accessor_fn(input.bitmask()),
                   size);
      else
        copy_range(p_out_b,
                   to_update,
                   bitmask_accessor_fn(target.bitmask()),
                   broadcast_fn<Bitmask::AllocType>(true),
                   size);
    }

    return ColumnView(TypeCode::STRING,
                      nullptr,
                      size,
                      p_out_b,
                      {ColumnView(TypeCode::INT32, p_out_o, size + 1),
                       ColumnView(TypeCode::INT8, p_out_c, num_chars)});
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::CAT32> * = nullptr>
  ColumnView operator()(const Rect<1> &target_bounds,
                        const Rect<1> &target_range,
                        const ColumnView &target,
                        const ColumnView &input,
                        alloc::Allocator &allocator)
  {
    assert(false);
    return ColumnView();
  }
};

struct BroadcastCopyRange {
  template <TypeCode CODE, std::enable_if_t<is_primitive_type<CODE>::value> * = nullptr>
  ColumnView operator()(const Rect<1> &target_bounds,
                        const Rect<1> &target_range,
                        const ColumnView &target,
                        const Scalar &input,
                        alloc::Allocator &allocator)
  {
    using VAL = pandas_type_of<CODE>;

    auto size  = target.size();
    auto p_out = allocator.allocate_elements<VAL>(size);

    auto to_update = compute_update_range(target_bounds, target_range);

    copy_range(
      p_out, to_update, accessor_fn<VAL>(target), broadcast_fn<VAL>(input.value<VAL>()), size);

    auto p_out_b = static_cast<Bitmask::AllocType *>(nullptr);
    if (target.nullable()) {
      p_out_b = allocator.allocate_elements<Bitmask::AllocType>(size);
      if (input.valid())
        copy_range(p_out_b,
                   to_update,
                   bitmask_accessor_fn(target.bitmask()),
                   broadcast_fn<Bitmask::AllocType>(true),
                   size);
      else
        copy_range(p_out_b,
                   to_update,
                   bitmask_accessor_fn(target.bitmask()),
                   broadcast_fn<Bitmask::AllocType>(false),
                   size);
    }

    return ColumnView(CODE, p_out, size, p_out_b);
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::STRING> * = nullptr>
  ColumnView operator()(const Rect<1> &target_bounds,
                        const Rect<1> &target_range,
                        const ColumnView &target,
                        const Scalar &scalar_input,
                        alloc::Allocator &allocator)
  {
    auto size  = target.size();
    auto input = scalar_input.valid() ? scalar_input.value<std::string>() : std::string("");

    auto to_update = compute_update_range(target_bounds, target_range);

    auto num_chars =
      sum(to_update, size_accessor_fn(target.child(0)), broadcast_fn<int32_t>(input.size()), size);

    auto p_out_o = allocator.allocate_elements<int32_t>(size + 1);
    auto p_out_c = allocator.allocate_elements<int8_t>(num_chars);

    copy_range(p_out_o,
               to_update,
               size_accessor_fn(target.child(0)),
               broadcast_fn<int32_t>(input.size()),
               size);
    p_out_o[size] = 0;
    thrust::exclusive_scan(thrust::host, p_out_o, p_out_o + size + 1, p_out_o);

    auto fn = make_range_scatter_fn(
      to_update, accessor_fn<std::string>(target), broadcast_fn<std::string>(input));

    int8_t *ptr = p_out_c;
    for (coord_t idx = 0; idx < size; ++idx) {
      std::string value = fn(idx);
      memcpy(ptr, value.c_str(), value.size());
      ptr += value.size();
    }
#ifdef DEBUG_PANDAS
    assert(ptr - p_out_c == num_chars);
#endif

    auto p_out_b = static_cast<Bitmask::AllocType *>(nullptr);
    if (target.nullable()) {
      p_out_b = allocator.allocate_elements<Bitmask::AllocType>(size);
      if (scalar_input.valid())
        copy_range(p_out_b,
                   to_update,
                   bitmask_accessor_fn(target.bitmask()),
                   broadcast_fn<Bitmask::AllocType>(true),
                   size);
      else
        copy_range(p_out_b,
                   to_update,
                   bitmask_accessor_fn(target.bitmask()),
                   broadcast_fn<Bitmask::AllocType>(false),
                   size);
    }

    return ColumnView(TypeCode::STRING,
                      nullptr,
                      size,
                      p_out_b,
                      {ColumnView(TypeCode::INT32, p_out_o, size + 1),
                       ColumnView(TypeCode::INT8, p_out_c, num_chars)});
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::CAT32> * = nullptr>
  ColumnView operator()(const Rect<1> &target_bounds,
                        const Rect<1> &target_range,
                        const ColumnView &target,
                        const Scalar &scalar_input,
                        alloc::Allocator &allocator)
  {
    assert(false);
    return ColumnView();
  }
};

ColumnView copy_range(const Rect<1> &target_bounds,
                      const Rect<1> &target_range,
                      const ColumnView &target,
                      const ColumnView &input,
                      alloc::Allocator &allocator)
{
  return type_dispatch(
    target.code(), CopyRange{}, target_bounds, target_range, target, input, allocator);
}

ColumnView copy_range(const Rect<1> &target_bounds,
                      const Rect<1> &target_range,
                      const ColumnView &target,
                      const Scalar &scalar_input,
                      alloc::Allocator &allocator)
{
  return type_dispatch(target.code(),
                       BroadcastCopyRange{},
                       target_bounds,
                       target_range,
                       target,
                       scalar_input,
                       allocator);
}

}  // namespace detail

/*static*/ void ScatterBySliceTask::cpu_variant(const Task *task,
                                                const std::vector<PhysicalRegion> &regions,
                                                Context context,
                                                Runtime *runtime)
{
  Deserializer ctx{task, regions};

  ScatterBySliceTaskArgs args;
  deserialize(ctx, args);

#ifdef DEBUG_PANDAS
  assert(args.requests.size() > 0);
#endif

  if (args.requests.front().target.empty()) {
    for (auto &req : args.requests) req.output.make_empty(true);
    return;
  }

  const auto &bounds = args.requests.front().target.shape();

  alloc::DeferredBufferAllocator allocator;
  for (auto &req : args.requests) {
    ColumnView result;
    if (args.input_is_scalar)
      result =
        detail::copy_range(bounds, args.range, req.target.view(), req.scalar_input, allocator);
    else
      result =
        detail::copy_range(bounds, args.range, req.target.view(), req.input.view(), allocator);

    req.output.return_from_view(allocator, result);
  }
}

void deserialize(Deserializer &ctx, ScatterBySliceTask::ScatterBySliceTaskArgs &args)
{
  deserialize(ctx, args.input_is_scalar);
  FromFuture<Rect<1>> range;
  deserialize(ctx, range);
  args.range = range.value();

  uint32_t num_values = 0;
  deserialize(ctx, num_values);
  for (uint32_t i = 0; i < num_values; ++i) {
    args.requests.push_back(ScatterBySliceArg{});
    ScatterBySliceArg &arg = args.requests.back();
    deserialize(ctx, arg.output);
    deserialize(ctx, arg.target);
    if (args.input_is_scalar)
      deserialize(ctx, arg.scalar_input);
    else
      deserialize(ctx, arg.input);
  }

#ifdef DEBUG_PANDAS
  args.sanity_check();
#endif
}

static void __attribute__((constructor)) register_tasks(void)
{
  ScatterBySliceTask::register_variants();
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

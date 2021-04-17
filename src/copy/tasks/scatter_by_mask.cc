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

#include "copy/tasks/scatter_by_mask.h"
#include "copy/copy.h"
#include "column/detail/column.h"
#include "util/allocator.h"
#include "util/type_dispatch.h"

namespace legate {
namespace pandas {
namespace copy {

using namespace Legion;

using ScatterByMaskArg = ScatterByMaskTask::ScatterByMaskTaskArgs::ScatterByMaskArg;

void ScatterByMaskTask::ScatterByMaskTaskArgs::sanity_check(void)
{
  for (auto &req : requests) {
    if (!input_is_scalar) assert(req.input.code() == req.target.code());
    assert(req.target.shape() == req.output.shape());
  }
}

using ColumnView = pandas::detail::Column;

namespace detail {

template <typename TargetAccessor, typename InputAccessor>
struct scatter_fn : public thrust::unary_function<coord_t, typename TargetAccessor::value_type> {
  scatter_fn(const std::vector<coord_t> &scatter_map,
             TargetAccessor &&target,
             InputAccessor &&input)
    : scatter_map_(scatter_map), target_(target), input_(input)
  {
  }

  typename TargetAccessor::value_type operator()(coord_t out_idx)
  {
    auto in_idx = scatter_map_[out_idx];
    return in_idx >= 0 ? input_(in_idx) : target_(out_idx);
  }

  const std::vector<coord_t> scatter_map_;
  TargetAccessor target_;
  InputAccessor input_;
};

template <typename TargetAccessor, typename InputAccessor>
decltype(auto) make_scatter_fn(const std::vector<coord_t> &scatter_map,
                               TargetAccessor &&target,
                               InputAccessor &&input)
{
  return scatter_fn<TargetAccessor, InputAccessor>(
    scatter_map, std::forward<TargetAccessor>(target), std::forward<InputAccessor>(input));
}

template <typename OutputIterator, typename TargetAccessor, typename InputAccessor>
void scatter_copy(OutputIterator &&output,
                  const std::vector<coord_t> &scatter_map,
                  TargetAccessor &&target,
                  InputAccessor &&input,
                  size_t size)
{
  auto fn = make_scatter_fn(
    scatter_map, std::forward<TargetAccessor>(target), std::forward<InputAccessor>(input));
  auto start = thrust::make_counting_iterator<coord_t>(0);
  auto stop  = thrust::make_counting_iterator<coord_t>(size);
  thrust::transform(thrust::host, start, stop, output, fn);
}

template <typename TargetAccessor, typename InputAccessor>
int32_t sum(const std::vector<coord_t> &scatter_map,
            TargetAccessor &&target,
            InputAccessor &&input,
            size_t size)
{
  auto fn = make_scatter_fn(
    scatter_map, std::forward<TargetAccessor>(target), std::forward<InputAccessor>(input));
  auto start = thrust::make_transform_iterator(thrust::make_counting_iterator<coord_t>(0), fn);
  auto stop  = thrust::make_transform_iterator(thrust::make_counting_iterator<coord_t>(size), fn);
  return thrust::reduce(thrust::host, start, stop);
}

struct Scatter {
  static Bitmask::AllocType *scatter_copy_bitmask(const std::vector<coord_t> &scatter_map,
                                                  const ColumnView &target,
                                                  const ColumnView &input,
                                                  alloc::Allocator &allocator,
                                                  size_t size)
  {
    if (!target.nullable()) return nullptr;
    auto p_out_b = allocator.allocate_elements<Bitmask::AllocType>(size);
    if (input.nullable())
      scatter_copy(p_out_b,
                   scatter_map,
                   bitmask_accessor_fn(target.bitmask()),
                   bitmask_accessor_fn(input.bitmask()),
                   size);
    else
      scatter_copy(p_out_b,
                   scatter_map,
                   bitmask_accessor_fn(target.bitmask()),
                   broadcast_fn<Bitmask::AllocType>(true),
                   size);
    return p_out_b;
  }

  template <TypeCode CODE, std::enable_if_t<is_primitive_type<CODE>::value> * = nullptr>
  ColumnView operator()(const std::vector<coord_t> &scatter_map,
                        const ColumnView &target,
                        const ColumnView &input,
                        alloc::Allocator &allocator)
  {
    using VAL = pandas_type_of<CODE>;

    auto size  = target.size();
    auto p_out = allocator.allocate_elements<VAL>(size);

    scatter_copy(p_out, scatter_map, accessor_fn<VAL>(target), accessor_fn<VAL>(input), size);

    auto p_out_b = scatter_copy_bitmask(scatter_map, target, input, allocator, size);

    return ColumnView(CODE, p_out, size, p_out_b);
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::STRING> * = nullptr>
  ColumnView operator()(const std::vector<coord_t> &scatter_map,
                        const ColumnView &target,
                        const ColumnView &input,
                        alloc::Allocator &allocator)
  {
    auto size = target.size();

    auto num_chars =
      sum(scatter_map, size_accessor_fn(target.child(0)), size_accessor_fn(input.child(0)), size);

    auto p_out_o = allocator.allocate_elements<int32_t>(size + 1);
    auto p_out_c = allocator.allocate_elements<int8_t>(num_chars);

    scatter_copy(p_out_o,
                 scatter_map,
                 size_accessor_fn(target.child(0)),
                 size_accessor_fn(input.child(0)),
                 size);
    p_out_o[size] = 0;
    thrust::exclusive_scan(thrust::host, p_out_o, p_out_o + size + 1, p_out_o);

    auto fn = make_scatter_fn(
      scatter_map, accessor_fn<std::string>(target), accessor_fn<std::string>(input));

    auto ptr = p_out_c;
    for (coord_t idx = 0; idx < size; ++idx) {
      std::string value = fn(idx);
      memcpy(ptr, value.c_str(), value.size());
      ptr += value.size();
    }
#ifdef DEBUG_PANDAS
    assert(ptr - p_out_c == num_chars);
#endif

    auto p_out_b = scatter_copy_bitmask(scatter_map, target, input, allocator, size);

    return ColumnView(TypeCode::STRING,
                      nullptr,
                      size,
                      p_out_b,
                      {ColumnView(TypeCode::INT32, p_out_o, size + 1),
                       ColumnView(TypeCode::INT8, p_out_c, num_chars)});
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::CAT32> * = nullptr>
  ColumnView operator()(const std::vector<coord_t> &scatter_map,
                        const ColumnView &target,
                        const ColumnView &input,
                        alloc::Allocator &allocator)
  {
    assert(false);
    return ColumnView();
  }
};

struct BroadcastScatter {
  static Bitmask::AllocType *scatter_copy_bitmask(const std::vector<coord_t> &scatter_map,
                                                  const ColumnView &target,
                                                  const Scalar &input,
                                                  alloc::Allocator &allocator,
                                                  size_t size)
  {
    if (!target.nullable()) return nullptr;
    auto p_out_b = allocator.allocate_elements<Bitmask::AllocType>(size);
    if (input.valid())
      scatter_copy(p_out_b,
                   scatter_map,
                   bitmask_accessor_fn(target.bitmask()),
                   broadcast_fn<Bitmask::AllocType>(true),
                   size);
    else
      scatter_copy(p_out_b,
                   scatter_map,
                   bitmask_accessor_fn(target.bitmask()),
                   broadcast_fn<Bitmask::AllocType>(false),
                   size);
    return p_out_b;
  }

  template <TypeCode CODE, std::enable_if_t<is_primitive_type<CODE>::value> * = nullptr>
  ColumnView operator()(const std::vector<coord_t> &scatter_map,
                        const ColumnView &target,
                        const Scalar &input,
                        alloc::Allocator &allocator)
  {
    using VAL = pandas_type_of<CODE>;

    auto size  = target.size();
    auto p_out = allocator.allocate_elements<VAL>(size);

    scatter_copy(
      p_out, scatter_map, accessor_fn<VAL>(target), broadcast_fn<VAL>(input.value<VAL>()), size);

    auto p_out_b = scatter_copy_bitmask(scatter_map, target, input, allocator, size);
    return ColumnView(CODE, p_out, size, p_out_b);
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::STRING> * = nullptr>
  ColumnView operator()(const std::vector<coord_t> &scatter_map,
                        const ColumnView &target,
                        const Scalar &scalar_input,
                        alloc::Allocator &allocator)
  {
    auto size  = target.size();
    auto input = scalar_input.valid() ? scalar_input.value<std::string>() : std::string("");

    auto num_chars = sum(
      scatter_map, size_accessor_fn(target.child(0)), broadcast_fn<int32_t>(input.size()), size);

    auto p_out_o = allocator.allocate_elements<int32_t>(size + 1);
    auto p_out_c = allocator.allocate_elements<int8_t>(num_chars);

    scatter_copy(p_out_o,
                 scatter_map,
                 size_accessor_fn(target.child(0)),
                 broadcast_fn<int32_t>(input.size()),
                 size);
    p_out_o[size] = 0;
    thrust::exclusive_scan(thrust::host, p_out_o, p_out_o + size + 1, p_out_o);

    auto fn = make_scatter_fn(
      scatter_map, accessor_fn<std::string>(target), broadcast_fn<std::string>(input));

    int8_t *ptr = p_out_c;
    for (coord_t idx = 0; idx < size; ++idx) {
      std::string value = fn(idx);
      memcpy(ptr, value.c_str(), value.size());
      ptr += value.size();
    }
#ifdef DEBUG_PANDAS
    assert(ptr - p_out_c == num_chars);
#endif

    auto p_out_b = scatter_copy_bitmask(scatter_map, target, scalar_input, allocator, size);

    return ColumnView(TypeCode::STRING,
                      nullptr,
                      size,
                      p_out_b,
                      {ColumnView(TypeCode::INT32, p_out_o, size + 1),
                       ColumnView(TypeCode::INT8, p_out_c, num_chars)});
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::CAT32> * = nullptr>
  ColumnView operator()(const std::vector<coord_t> &scatter_map,
                        const ColumnView &target,
                        const Scalar &scalar_input,
                        alloc::Allocator &allocator)
  {
    assert(false);
    return ColumnView();
  }
};

ColumnView boolean_mask_scatter(const std::vector<coord_t> &scatter_map,
                                const ColumnView &target,
                                const ColumnView &input,
                                alloc::Allocator &allocator)
{
  return type_dispatch(target.code(), Scatter{}, scatter_map, target, input, allocator);
}

ColumnView boolean_mask_scatter(const std::vector<coord_t> &scatter_map,
                                const ColumnView &target,
                                const Scalar &input,
                                alloc::Allocator &allocator)
{
  return type_dispatch(target.code(), BroadcastScatter{}, scatter_map, target, input, allocator);
}

}  // namespace detail

/*static*/ void ScatterByMaskTask::cpu_variant(const Task *task,
                                               const std::vector<PhysicalRegion> &regions,
                                               Context context,
                                               Runtime *runtime)
{
  Deserializer ctx{task, regions};

  ScatterByMaskTaskArgs args;
  deserialize(ctx, args);

  if (args.mask.empty()) {
    for (auto &req : args.requests) req.output.make_empty(true);
    return;
  }

  auto mask = args.mask.view();
  alloc::DeferredBufferAllocator allocator;

  auto size = mask.size();
  std::vector<coord_t> scatter_map(size, 0);
  coord_t offset = 0;
  if (mask.nullable()) {
    auto mask_b = mask.bitmask();
    for (coord_t idx = 0; idx < size; ++idx) {
      bool valid       = mask_b.get(idx) && mask.element<bool>(idx);
      scatter_map[idx] = valid ? offset : -1;
      offset += static_cast<coord_t>(valid);
    }
  } else
    for (coord_t idx = 0; idx < size; ++idx) {
      bool valid       = mask.element<bool>(idx);
      scatter_map[idx] = valid ? offset : -1;
      offset += static_cast<coord_t>(valid);
    }

  for (auto &req : args.requests) {
    ColumnView result;
    if (args.input_is_scalar)
      result =
        detail::boolean_mask_scatter(scatter_map, req.target.view(), req.scalar_input, allocator);
    else
      result =
        detail::boolean_mask_scatter(scatter_map, req.target.view(), req.input.view(), allocator);
    req.output.return_from_view(allocator, result);
  }
}

void deserialize(Deserializer &ctx, ScatterByMaskTask::ScatterByMaskTaskArgs &args)
{
  deserialize(ctx, args.input_is_scalar);
  deserialize(ctx, args.mask);

  uint32_t num_values = 0;
  deserialize(ctx, num_values);
  for (uint32_t i = 0; i < num_values; ++i) {
    args.requests.push_back(ScatterByMaskArg{});
    ScatterByMaskArg &arg = args.requests.back();
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
  ScatterByMaskTask::register_variants();
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

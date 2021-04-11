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

#include <cmath>

#include "bitmask/tasks/init_bitmask.h"
#include "column/column.h"
#include "column/device_column.h"
#include "cudf_util/allocators.h"
#include "util/gpu_task_context.h"
#include "util/type_dispatch.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_device_view.cuh>

#include <rmm/exec_policy.hpp>

namespace legate {
namespace pandas {
namespace bitmask {

using namespace Legion;

namespace detail {

template <class T>
struct is_valid_fn {
  is_valid_fn(cudf::column_device_view in, T value) : in_(in), value_(value) {}

  template <class _T = T, std::enable_if_t<std::is_integral<_T>::value> * = nullptr>
  __device__ constexpr Bitmask::AllocType operator()(const size_t idx) const
  {
    return static_cast<Bitmask::AllocType>(in_.element<_T>(idx) != value_);
  }

  template <class _T = T, std::enable_if_t<!std::is_integral<_T>::value> * = nullptr>
  __device__ constexpr Bitmask::AllocType operator()(const size_t idx) const
  {
    return static_cast<Bitmask::AllocType>(!std::isnan(in_.element<_T>(idx)));
  }

  cudf::column_device_view in_;
  T value_;
};

struct Initializer {
  template <TypeCode CODE>
  std::unique_ptr<cudf::column> operator()(cudf::column_view &&input,
                                           const Scalar &null_value,
                                           cudaStream_t stream,
                                           rmm::mr::device_memory_resource *mr)
  {
    using VAL = pandas_type_of<CODE>;

    auto size   = input.size();
    auto result = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::UINT8}, size, cudf::mask_state::UNALLOCATED, stream, mr);
    auto m_result = result->mutable_view();

    auto d_input = cudf::column_device_view::create(input, stream);
    auto val     = null_value.value<VAL>();

    is_valid_fn<VAL> is_valid{*d_input, val};
    thrust::transform(rmm::exec_policy(stream),
                      thrust::counting_iterator<size_t>(0),
                      thrust::counting_iterator<size_t>(size),
                      m_result.begin<Bitmask::AllocType>(),
                      is_valid);

    return result;
  }
};

std::unique_ptr<cudf::column> initialize_bitmask(cudf::column_view &&input,
                                                 const Scalar &null_value,
                                                 cudaStream_t stream,
                                                 rmm::mr::device_memory_resource *mr)
{
  return type_dispatch_numeric_only(
    null_value.code(), Initializer{}, std::move(input), null_value, stream, mr);
}

}  // namespace detail

/*static*/ void InitBitmaskTask::gpu_variant(const Task *task,
                                             const std::vector<PhysicalRegion> &regions,
                                             Context context,
                                             Runtime *runtime)
{
  Deserializer ctx{task, regions};

  Scalar null_value;
  OutputColumn bitmask;
  Column<true> input;
  deserialize(ctx, null_value);
  deserialize(ctx, bitmask);
  deserialize(ctx, input);

  if (input.empty()) {
    bitmask.make_empty();
    return;
  }

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  DeferredBufferAllocator mr;

  auto input_view = DeviceColumn<true>(input).to_cudf_column(stream);
  auto result     = detail::initialize_bitmask(std::move(input_view), null_value, stream, &mr);
  DeviceOutputColumn(bitmask).return_from_cudf_column(mr, result->view(), stream);
}

}  // namespace bitmask
}  // namespace pandas
}  // namespace legate

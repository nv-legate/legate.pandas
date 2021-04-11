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

#include "copy/tasks/write_at.h"
#include "column/column.h"
#include "column/device_column.h"
#include "cudf_util/allocators.h"
#include "cudf_util/scalar.h"
#include "util/gpu_task_context.h"
#include "deserializer.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy.hpp>

#include <rmm/exec_policy.hpp>

namespace legate {
namespace pandas {
namespace copy {

namespace detail {
namespace gpu {

struct gen_cond_fn {
  gen_cond_fn(coord_t to_find) : to_find_(to_find) {}

  __device__ bool operator()(coord_t idx) const { return idx != to_find_; }

  coord_t to_find_;
};

}  // namespace gpu
}  // namespace detail

using namespace Legion;

/*static*/ void WriteAtTask::gpu_variant(const Task *task,
                                         const std::vector<PhysicalRegion> &regions,
                                         Context context,
                                         Runtime *runtime)
{
  Deserializer ctx{task, regions};

  FromFuture<int64_t> idx_fut;
  OutputColumn output;
  Column<true> input;
  Scalar value;
  deserialize(ctx, idx_fut);
  deserialize(ctx, output);
  deserialize(ctx, input);
  deserialize(ctx, value);

  auto idx   = idx_fut.value();
  auto shape = input.shape();

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  DeferredBufferAllocator mr;

  auto size = input.num_elements();
  auto cond = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::BOOL8}, size, cudf::mask_state::UNALLOCATED, stream, &mr);
  auto m_cond = cond->mutable_view();
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<coord_t>(0),
                    thrust::make_counting_iterator<coord_t>(size),
                    m_cond.begin<bool>(),
                    detail::gpu::gen_cond_fn{idx - shape.lo[0]});

  auto input_view = DeviceColumn<true>{input}.to_cudf_column(stream);
  auto value_sc   = to_cudf_scalar(value.valid() ? value.raw_ptr() : nullptr, input.code(), stream);
  auto result     = cudf::detail::copy_if_else(input_view, *value_sc, cond->view(), stream, &mr);
  DeviceOutputColumn{output}.return_from_cudf_column(mr, result->view(), stream);
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

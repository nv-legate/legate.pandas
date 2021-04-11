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

#include "index/tasks/find_bounds.h"
#include "index/search.cuh"
#include "column/device_column.h"
#include "cudf_util/allocators.h"
#include "cudf_util/detail.h"
#include "cudf_util/scalar.h"
#include "cudf_util/types.h"
#include "util/allocator.h"
#include "util/gpu_task_context.h"

#include <cudf/column/column_view.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/binaryop.hpp>

#include <rmm/exec_policy.hpp>

namespace legate {
namespace pandas {
namespace index {

using namespace Legion;

namespace detail {

coord_t search_forward(cudf::column_view in,
                       const Scalar &to_find,
                       const Rect<1> &bounds,
                       int32_t total_volume,
                       cudaStream_t stream)
{
  auto result = search(in, to_find, true, stream);
  if (!result.valid())
    return total_volume;
  else
    return result.value<int64_t>() + bounds.lo[0];
}

coord_t search_backward(cudf::column_view in,
                        const Scalar &to_find,
                        const Rect<1> &bounds,
                        cudaStream_t stream)
{
  auto result = search(in, to_find, false, stream);
  if (!result.valid())
    return -1;
  else
    return result.value<int64_t>() + bounds.lo[0];
}

}  // namespace detail

/*static*/ Rect<1> FindBoundsTask::gpu_variant(const Task *task,
                                               const std::vector<PhysicalRegion> &regions,
                                               Context context,
                                               Runtime *runtime)
{
  Deserializer ctx{task, regions};

  Scalar start;
  Scalar stop;
  FromFuture<int64_t> volume;
  Column<true> column;

  deserialize(ctx, start);
  deserialize(ctx, stop);
  deserialize(ctx, volume);
  deserialize(ctx, column);

  const auto &bounds = column.shape();

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  std::vector<cudf::column_view> columns;
  columns.push_back(DeviceColumn<true>(column).to_cudf_column(stream));

#ifdef DEBUG_PANDAS
  // FIXME: We need to support multi-index
  assert(columns.size() == 1);
#endif

  auto lo =
    start.valid() ? detail::search_forward(columns[0], start, bounds, volume.value(), stream) : 0;
  auto hi =
    stop.valid() ? detail::search_backward(columns[0], stop, bounds, stream) : volume.value() - 1;

  return Rect<1>{Point<1>{lo}, Point<1>{hi}};
}

}  // namespace index
}  // namespace pandas
}  // namespace legate

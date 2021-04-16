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

#include "transform/tasks/notna.h"
#include "column/column.h"
#include "cudf_util/allocators.h"
#include "cudf_util/column.h"
#include "util/gpu_task_context.h"
#include "deserializer.h"

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/unary.hpp>

namespace legate {
namespace pandas {
namespace transform {

using namespace Legion;

/*static*/ void NotNaTask::gpu_variant(const Task *task,
                                       const std::vector<PhysicalRegion> &regions,
                                       Context context,
                                       Runtime *runtime)
{
  Deserializer ctx{task, regions};

  OutputColumn out;
  Column<true> in;

  deserialize(ctx, out);
  deserialize(ctx, in);

  if (in.empty()) {
    out.make_empty(true);
    return;
  }

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  auto input = to_cudf_column(in, stream);

  DeferredBufferAllocator mr;

  auto input_device_view = cudf::column_device_view::create(input);
  auto device_view       = *input_device_view;
  auto predicate = [device_view] __device__(auto index) { return (device_view.is_valid(index)); };
  auto result    = cudf::detail::true_if(thrust::make_counting_iterator(0),
                                      thrust::make_counting_iterator(input.size()),
                                      input.size(),
                                      predicate,
                                      stream,
                                      &mr);

  from_cudf_column(out, std::move(result), stream, mr);
}

}  // namespace transform
}  // namespace pandas
}  // namespace legate

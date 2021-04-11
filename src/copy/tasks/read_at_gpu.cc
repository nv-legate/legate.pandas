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

#include "copy/tasks/read_at.h"
#include "column/column.h"
#include "column/device_column.h"
#include "cudf_util/allocators.h"
#include "util/gpu_task_context.h"
#include "deserializer.h"

#include <cudf/detail/copy.hpp>

namespace legate {
namespace pandas {
namespace copy {

using namespace Legion;

/*static*/ int64_t ReadAtTask::gpu_variant(const Task *task,
                                           const std::vector<PhysicalRegion> &regions,
                                           Context context,
                                           Runtime *runtime)
{
  Deserializer ctx{task, regions};

  FromFuture<int64_t> idx_fut;
  OutputColumn output;
  Column<true> input;
  deserialize(ctx, idx_fut);
  deserialize(ctx, output);
  deserialize(ctx, input);

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  auto idx   = idx_fut.value();
  auto shape = input.shape();

  if (idx < shape.lo[0] || shape.hi[0] < idx) {
    output.make_empty(true);
    return 0;
  } else {
    DeferredBufferAllocator mr;
    auto cudf_input = DeviceColumn<true>{input}.to_cudf_column(stream);
    auto pos        = idx - shape.lo[0];
    auto sliced     = cudf::detail::slice(cudf_input, pos, pos + 1);
    cudf::column copy(sliced, stream, &mr);
    DeviceOutputColumn{output}.return_from_cudf_column(mr, copy.view(), stream);
    return 1;
  }
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

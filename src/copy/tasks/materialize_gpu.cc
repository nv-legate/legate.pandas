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

#include "copy/tasks/materialize.h"
#include "copy/materialize.cuh"

#include "column/column.h"
#include "column/device_column.h"
#include "cudf_util/allocators.h"
#include "util/gpu_task_context.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace copy {

using namespace Legion;

/*static*/ void MaterializeTask::gpu_variant(const Task *task,
                                             const std::vector<PhysicalRegion> &regions,
                                             Context context,
                                             Runtime *runtime)
{
  Deserializer ctx{task, regions};

  FromFuture<int64_t> start;
  FromFuture<int64_t> step;
  OutputColumn out;
  deserialize(ctx, start);
  deserialize(ctx, step);
  deserialize(ctx, out);

  auto shape = out.shape();
  if (shape.empty()) {
    out.make_empty();
    return;
  }

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  DeferredBufferAllocator mr;
  auto result = materialize(shape, start.value(), step.value(), stream, &mr);
  DeviceOutputColumn{out}.return_from_cudf_column(mr, result->view(), stream);
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

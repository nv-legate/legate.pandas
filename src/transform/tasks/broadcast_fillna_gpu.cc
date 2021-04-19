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

#include "transform/tasks/broadcast_fillna.h"
#include "column/column.h"
#include "cudf_util/allocators.h"
#include "cudf_util/column.h"
#include "cudf_util/scalar.h"
#include "util/gpu_task_context.h"
#include "deserializer.h"

#include <cudf/detail/replace.hpp>

namespace legate {
namespace pandas {
namespace transform {

using namespace Legion;

/*static*/ void BroadcastFillNaTask::gpu_variant(const Task *task,
                                                 const std::vector<PhysicalRegion> &regions,
                                                 Context context,
                                                 Runtime *runtime)
{
  Deserializer ctx{task, regions};

  OutputColumn out;
  Column<true> in;
  Scalar scalar;

  deserialize(ctx, out);
  deserialize(ctx, in);
  deserialize(ctx, scalar);

  if (in.empty()) {
    out.make_empty(true);
    return;
  }

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  auto input = to_cudf_column(in, stream);
  std::unique_ptr<cudf::scalar> p_fill_value =
    to_cudf_scalar(scalar.raw_ptr(), scalar.code(), stream);

  DeferredBufferAllocator mr;
  auto result = cudf::detail::replace_nulls(input, *p_fill_value, stream, &mr);
  from_cudf_column(out, std::move(result), stream, mr);
}

}  // namespace transform
}  // namespace pandas
}  // namespace legate

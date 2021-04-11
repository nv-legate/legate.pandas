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

#include <math.h>

#include "bitmask/tasks/to_bitmask.h"
#include "bitmask/bitmask.h"
#include "column/column.h"
#include "cudf_util/bitmask.h"
#include "util/gpu_task_context.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace bitmask {

using namespace Legion;

/*static*/ void ToBitmaskTask::gpu_variant(const Task *task,
                                           const std::vector<PhysicalRegion> &regions,
                                           Context context,
                                           Runtime *runtime)
{
  Deserializer ctx{task, regions};

  OutputColumn out;
  Column<true> in;

  deserialize(ctx, out);
  deserialize(ctx, in);

  auto size = in.num_elements();
  out.allocate(size);
  if (size == 0) {
    out.make_empty();
    return;
  }

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  util::to_bitmask(
    out.raw_column<Bitmask::AllocType>(), in.raw_column_read<Bitmask::AllocType>(), size, stream);
}

}  // namespace bitmask
}  // namespace pandas
}  // namespace legate

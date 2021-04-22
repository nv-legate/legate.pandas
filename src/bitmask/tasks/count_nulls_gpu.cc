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

#include "bitmask/tasks/count_nulls.h"
#include "bitmask/bitmask.h"
#include "util/gpu_task_context.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace bitmask {

using namespace Legion;

/*static*/ uint64_t CountNullsTask::gpu_variant(const Task *task,
                                                const std::vector<PhysicalRegion> &regions,
                                                Context context,
                                                Runtime *runtime)
{
  Deserializer ctx{task, regions};

  Column<true> column;
  deserialize(ctx, column);

  if (column.num_elements() == 0) return 0;

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  const Bitmask bitmask(column.raw_column_read<Bitmask::AllocType>(), column.num_elements());
  return bitmask.count_unset_bits(stream);
}

}  // namespace bitmask
}  // namespace pandas
}  // namespace legate

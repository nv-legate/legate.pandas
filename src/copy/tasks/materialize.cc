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

#include "copy/materialize.h"
#include "copy/tasks/materialize.h"

#include "column/column.h"
#include "column/detail/column.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace copy {

using namespace Legion;

/*static*/ void MaterializeTask::cpu_variant(const Task *task,
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

  alloc::DeferredBufferAllocator allocator;
  auto shape        = out.shape();
  auto materialized = materialize(shape, start.value(), step.value(), allocator);
  out.return_from_view(allocator, materialized);
}

static void __attribute__((constructor)) register_tasks(void)
{
  MaterializeTask::register_variants();
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

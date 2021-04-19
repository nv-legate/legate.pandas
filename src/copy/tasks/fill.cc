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

#include "copy/tasks/fill.h"
#include "copy/fill.h"

namespace legate {
namespace pandas {
namespace copy {

using namespace Legion;

/*static*/ void FillTask::cpu_variant(const Task *task,
                                      const std::vector<PhysicalRegion> &regions,
                                      Context context,
                                      Runtime *runtime)
{
  Deserializer ctx{task, regions};

  FromFuture<int64_t> volume;
  deserialize(ctx, volume);
  OutputColumn out;
  deserialize(ctx, out);
  int32_t num_pieces;
  deserialize(ctx, num_pieces);
  Scalar value;
  deserialize(ctx, value);

  int64_t total_size = volume.value();
  int64_t task_id    = task->index_point[0];
  int64_t my_size    = (task_id + 1) * total_size / num_pieces - task_id * total_size / num_pieces;

#ifdef DEBUG_PANDAS
  assert(my_size >= 0);
#endif

  if (my_size == 0) {
    out.make_empty(true);
    return;
  }

  alloc::DeferredBufferAllocator allocator;
  auto filled = fill(value, static_cast<size_t>(my_size), allocator);
  out.return_from_view(allocator, filled);
}

static void __attribute__((constructor)) register_tasks(void) { FillTask::register_variants(); }

}  // namespace copy
}  // namespace pandas
}  // namespace legate

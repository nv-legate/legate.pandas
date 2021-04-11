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
#include "copy/gather.h"
#include "util/allocator.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace copy {

using namespace Legion;

/*static*/ int64_t ReadAtTask::cpu_variant(const Task *task,
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

  auto idx   = idx_fut.value();
  auto shape = input.shape();

  if (idx < shape.lo[0] || shape.hi[0] < idx) {
    output.make_empty(true);
    return 0;
  } else {
    alloc::DeferredBufferAllocator allocator;
    auto result =
      gather(input.view(), {idx - shape.lo[0]}, false, OutOfRangePolicy::IGNORE, allocator);
    output.return_from_view(allocator, result);
    return 1;
  }
}

static void __attribute__((constructor)) register_tasks(void)
{
  ReadAtTask::register_variants_with_return<int64_t, int64_t>();
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

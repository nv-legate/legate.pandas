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

#include "copy/tasks/write_at.h"
#include "column/column.h"
#include "copy/copy_if_else.h"
#include "copy/fill.h"
#include "util/allocator.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace copy {

using namespace Legion;

/*static*/ void WriteAtTask::cpu_variant(const Task *task,
                                         const std::vector<PhysicalRegion> &regions,
                                         Context context,
                                         Runtime *runtime)
{
  Deserializer ctx{task, regions};

  FromFuture<int64_t> idx_fut;
  OutputColumn output;
  Column<true> input;
  Scalar value;
  deserialize(ctx, idx_fut);
  deserialize(ctx, output);
  deserialize(ctx, input);
  deserialize(ctx, value);

  auto idx   = idx_fut.value();
  auto shape = input.shape();

  std::vector<uint8_t> cond_vec(input.num_elements(), static_cast<uint8_t>(true));
  if (shape.lo[0] <= idx && idx <= shape.hi[0]) cond_vec[idx - shape.lo[0]] = false;
  detail::Column cond(TypeCode::BOOL, cond_vec.data(), cond_vec.size());

  alloc::DeferredBufferAllocator allocator;
  auto result = copy_if_else(cond, input.view(), value, false, allocator);
  output.return_from_view(allocator, result);
}

static void __attribute__((constructor)) register_tasks(void) { WriteAtTask::register_variants(); }

}  // namespace copy
}  // namespace pandas
}  // namespace legate

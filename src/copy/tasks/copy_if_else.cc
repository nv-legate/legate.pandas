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

#include "copy/tasks/copy_if_else.h"
#include "copy/copy_if_else.h"
#include "column/column.h"

namespace legate {
namespace pandas {
namespace copy {

using namespace Legion;

/*static*/ void CopyIfElseTask::cpu_variant(const Task *task,
                                            const std::vector<PhysicalRegion> &regions,
                                            Context context,
                                            Runtime *runtime)
{
  Deserializer ctx{task, regions};

  OutputColumn out;
  Column<true> input;
  Column<true> cond;

  deserialize(ctx, out);
  deserialize(ctx, input);
  deserialize(ctx, cond);

  bool negate{false};
  bool has_other{false};
  bool other_is_scalar{true};
  Column<true> other{};
  Scalar other_scalar{};

  deserialize(ctx, negate);
  deserialize(ctx, has_other);
  if (has_other) {
    deserialize(ctx, other_is_scalar);
    if (other_is_scalar)
      deserialize(ctx, other_scalar);
    else
      deserialize(ctx, other);
  }

  auto size = input.num_elements();
  if (size == 0) {
    out.make_empty(true);
    return;
  }

  alloc::DeferredBufferAllocator allocator;
  if (has_other)
    if (other_is_scalar) {
      auto result = copy_if_else(cond.view(), input.view(), other_scalar, negate, allocator);
      out.return_from_view(allocator, result);
    } else {
      auto result = copy_if_else(cond.view(), input.view(), other.view(), negate, allocator);
      out.return_from_view(allocator, result);
    }
  else {
    auto result = copy_if_else(cond.view(), input.view(), negate, allocator);
    out.return_from_view(allocator, result);
  }
}

static void __attribute__((constructor)) register_tasks(void)
{
  CopyIfElseTask::register_variants();
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

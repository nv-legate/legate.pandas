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

#include "transform/tasks/isna.h"
#include "column/column.h"
#include "util/allocator.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace transform {

using namespace Legion;
using ColumnView = detail::Column;

namespace detail {

ColumnView isna(const ColumnView &in, alloc::Allocator &allocator)
{
  auto size = in.size();
  auto in_b = in.bitmask();
  auto out  = allocator.allocate_elements<bool>(size);
  for (auto idx = 0; idx < size; ++idx) out[idx] = !in_b.get(idx);
  return ColumnView(TypeCode::BOOL, out, size);
}

}  // namespace detail

/*static*/ void IsNaTask::cpu_variant(const Task *task,
                                      const std::vector<PhysicalRegion> &regions,
                                      Context context,
                                      Runtime *runtime)
{
  Deserializer ctx{task, regions};

  OutputColumn out;
  Column<true> in;

  deserialize(ctx, out);
  deserialize(ctx, in);

  if (in.empty()) {
    out.make_empty(true);
    return;
  }

  alloc::DeferredBufferAllocator allocator;
  auto result = detail::isna(in.view(), allocator);
  out.return_from_view(allocator, result);
}

static void __attribute__((constructor)) register_tasks(void) { IsNaTask::register_variants(); }

}  // namespace transform
}  // namespace pandas
}  // namespace legate

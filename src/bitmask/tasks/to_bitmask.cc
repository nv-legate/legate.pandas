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
#include "bitmask/compact_bitmask.h"
#include "column/column.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace bitmask {

using namespace Legion;

/*static*/ void ToBitmaskTask::cpu_variant(const Task *task,
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

  CompactBitmask bitmask(out.raw_column<CompactBitmask::AllocType>(), size);
  Bitmask boolmask(in.raw_column_read<Bitmask::AllocType>(), size);
  for (auto idx = 0; idx < size; ++idx) bitmask.set(idx, boolmask.get(idx));
}

static void __attribute__((constructor)) register_tasks(void)
{
  ToBitmaskTask::register_variants();
}

}  // namespace bitmask
}  // namespace pandas
}  // namespace legate

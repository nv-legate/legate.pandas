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

#include "string/tasks/import_offsets.h"
#include "bitmask/bitmask.h"
#include "bitmask/compact_bitmask.h"
#include "column/column.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace string {

using namespace Legion;

/*static*/ void ImportOffsetsTask::cpu_variant(const Task *task,
                                               const std::vector<PhysicalRegion> &regions,
                                               Context context,
                                               Runtime *runtime)
{
  Deserializer ctx{task, regions};

  Column<false> out;
  RegionArg<true> in_offsets_arg;

  deserialize(ctx, out);
  deserialize(ctx, in_offsets_arg);

  auto *out_offsets = out.child(0).raw_column_write<int32_t>();
  auto *in_offsets  = in_offsets_arg.raw_read<int32_t>();

  const auto num_offsets = out.child(0).num_elements();
  assert(in_offsets_arg.size() == num_offsets);

  for (size_t i = 0; i < num_offsets; ++i) out_offsets[i] = in_offsets[i];

  if (out.nullable()) {
    const auto size = num_offsets - 1;
    RegionArg<true> in_bitmask_arg;
    deserialize(ctx, in_bitmask_arg);

    assert(in_bitmask_arg.size() == (size + 7) / 8);

    CompactBitmask in_b(in_bitmask_arg.raw_read<CompactBitmask::AllocType>(), size);
    Bitmask out_b = out.write_bitmask();
    for (size_t i = 0; i < size; ++i) out_b.set(i, in_b.get(i));
  }
}

static void __attribute__((constructor)) register_tasks(void)
{
  ImportOffsetsTask::register_variants();
}

}  // namespace string
}  // namespace pandas
}  // namespace legate

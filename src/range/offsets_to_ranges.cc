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

#include "range/offsets_to_ranges.h"
#include "column/column.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace range {

using namespace Legion;

/*static*/ void OffsetsToRangesTask::cpu_variant(const Task *task,
                                                 const std::vector<PhysicalRegion> &regions,
                                                 Context context,
                                                 Runtime *runtime)
{
  Deserializer ctx{task, regions};

  Column<false> out;
  Column<true> in;
  Column<true> chars;

  deserialize(ctx, out);
  deserialize(ctx, in);
  deserialize(ctx, chars);

  const auto num_offsets = in.num_elements();

  if (num_offsets == 0) return;

#ifdef DEBUG_PANDAS
  assert(num_offsets > 1);
#endif

  const auto num_ranges = num_offsets - 1;

  auto *out_ranges = out.raw_column_write<Rect<1>>();
  auto *in_offsets = in.raw_column_read<int32_t>();
  auto offset      = chars.shape().lo[0];
  for (auto i = 0; i < num_ranges; ++i) {
    coord_t lo    = in_offsets[i] + offset;
    coord_t hi    = in_offsets[i + 1] - 1 + offset;
    out_ranges[i] = Rect<1>{lo, hi};
  }
}

static void __attribute__((constructor)) register_tasks(void)
{
  OffsetsToRangesTask::register_variants();
}

}  // namespace range
}  // namespace pandas
}  // namespace legate

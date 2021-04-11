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

#include "range/ranges_to_offsets.h"
#include "column/column.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace range {

using namespace Legion;

/*static*/ int64_t RangesToOffsetsTask::cpu_variant(const Task *task,
                                                    const std::vector<PhysicalRegion> &regions,
                                                    Context context,
                                                    Runtime *runtime)
{
  Deserializer ctx{task, regions};

  OutputColumn out;
  Column<true> in;

  deserialize(ctx, out);
  deserialize(ctx, in);

  int64_t size = static_cast<int64_t>(in.num_elements());
  if (size == 0) {
    out.allocate(0);
    return 0;
  }

  out.allocate(size + 1);

  auto *out_offsets = out.raw_column<int32_t>();
  auto *in_ranges   = in.raw_column_read<Rect<1>>();

  auto offset = 0;
  for (auto i = 0; i < size; ++i) {
    auto &r        = in_ranges[i];
    out_offsets[i] = offset;
    offset += r.hi[0] - r.lo[0] + 1;
  }
  out_offsets[size] = offset;

  return size + 1;
}

static void __attribute__((constructor)) register_tasks(void)
{
  RangesToOffsetsTask::register_variants_with_return<int64_t, int64_t>();
}

}  // namespace range
}  // namespace pandas
}  // namespace legate

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

#include <limits>

#include "index/tasks/find_bounds_in_range.h"
#include "index/range_index.h"
#include "scalar/scalar.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace index {

using namespace Legion;

/*static*/ Rect<1> FindBoundsInRangeTask::cpu_variant(const Task *task,
                                                      const std::vector<PhysicalRegion> &regions,
                                                      Context context,
                                                      Runtime *runtime)
{
  Deserializer ctx{task, regions};

  RangeIndex index;
  Scalar start_sc;
  Scalar stop_sc;

  deserialize(ctx, index);
  deserialize(ctx, start_sc);
  deserialize(ctx, stop_sc);

  auto start = start_sc.valid() ? start_sc.value<int64_t>() : index.start;
  // Make an open interval when the slice has no upper bound
  auto stop = stop_sc.valid() ? stop_sc.value<int64_t>() : index.stop;

  auto rect = index.inverse_image(std::make_pair(start, stop), false);
  // Make sure to convert the range to a closed one when the slice had no upper bound;
  rect.hi[0] -= !stop_sc.valid();
  return rect;
}

static void __attribute__((constructor)) register_tasks(void)
{
  FindBoundsInRangeTask::register_variants_with_return<Rect<1>, Rect<1>>();
}

}  // namespace index
}  // namespace pandas
}  // namespace legate

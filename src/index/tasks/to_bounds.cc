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

#include "index/tasks/to_bounds.h"
#include "scalar/scalar.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace index {

using namespace Legion;

/*static*/ Rect<1> ToBoundsTask::cpu_variant(const Task *task,
                                             const std::vector<PhysicalRegion> &regions,
                                             Context context,
                                             Runtime *runtime)
{
  Deserializer ctx{task, regions};

  FromFuture<int64_t> volume;
  Scalar start;
  Scalar stop;

  deserialize(ctx, volume);
  deserialize(ctx, start);
  deserialize(ctx, stop);

  auto lo = start.valid() ? start.value<int64_t>() : static_cast<int64_t>(0);
  if (lo < 0) lo += volume.value();
  auto hi = stop.valid() ? stop.value<int64_t>() : static_cast<int64_t>(volume.value());
  if (hi < 0) hi += volume.value();
  hi -= 1;

  return Rect<1>{Point<1>{lo}, Point<1>{hi}};
}

static void __attribute__((constructor)) register_tasks(void)
{
  ToBoundsTask::register_variants_with_return<Rect<1>, Rect<1>>();
}

}  // namespace index
}  // namespace pandas
}  // namespace legate

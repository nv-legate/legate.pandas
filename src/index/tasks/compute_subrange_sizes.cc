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

#include "index/tasks/compute_subrange_sizes.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace index {

using namespace Legion;

/*static*/ int64_t ComputeSubrangeSizesTask::cpu_variant(const Task *task,
                                                         const std::vector<PhysicalRegion> &regions,
                                                         Context context,
                                                         Runtime *runtime)
{
  Deserializer ctx{task, regions};

  Rect<1> bounds;
  deserialize_from_future(ctx, bounds);

  Rect<1> subrect(regions[0]);

  return static_cast<int64_t>(subrect.intersection(bounds).volume());
}

static void __attribute__((constructor)) register_tasks(void)
{
  ComputeSubrangeSizesTask::register_variants_with_return<int64_t, int64_t>();
}

}  // namespace index
}  // namespace pandas
}  // namespace legate

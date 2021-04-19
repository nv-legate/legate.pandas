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

#include "util/to_column.h"
#include "column/column.h"
#include "scalar/scalar.h"
#include "util/zip_for_each.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace util {

using namespace Legion;

/*static*/ void ToColumnTask::cpu_variant(const Task *task,
                                          const std::vector<PhysicalRegion> &regions,
                                          Context context,
                                          Runtime *runtime)
{
  Deserializer ctx{task, regions};

  OutputColumn output;
  deserialize(ctx, output);

  auto shape = output.shape();

  if (shape.volume() == 0)
    output.make_empty(true);
  else {
    std::vector<Scalar> scalars;
    for (auto idx = shape.lo[0]; idx <= shape.hi[0]; ++idx) {
      auto scalar = task->futures[idx].get_result<Scalar>();
      scalars.push_back(std::move(scalar));
    }
    output.return_from_scalars(scalars);
  }
}

static void __attribute__((constructor)) register_tasks(void) { ToColumnTask::register_variants(); }

}  // namespace util
}  // namespace pandas
}  // namespace legate

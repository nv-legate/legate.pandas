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

#include "copy/tasks/densify.h"
#include "column/column.h"
#include "util/zip_for_each.h"

namespace legate {
namespace pandas {
namespace copy {

using namespace Legion;

/*static*/ int64_t DensifyTask::cpu_variant(const Task *task,
                                            const std::vector<PhysicalRegion> &regions,
                                            Context context,
                                            Runtime *runtime)
{
  Deserializer ctx{task, regions};

  std::vector<Column<true>> inputs;
  std::vector<OutputColumn> outputs;

  deserialize(ctx, inputs);
  deserialize(ctx, outputs);

  util::for_each(outputs, inputs, [&](auto &output, auto &input) { output.copy(input, true); });

  return static_cast<int64_t>(inputs[0].num_elements());
}

static void __attribute__((constructor)) register_tasks(void)
{
  DensifyTask::register_variants_with_return<int64_t, int64_t>();
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

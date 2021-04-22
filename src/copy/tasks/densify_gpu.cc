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
#include "cudf_util/allocators.h"
#include "cudf_util/column.h"
#include "util/gpu_task_context.h"

#include <cudf/table/table.hpp>

namespace legate {
namespace pandas {
namespace copy {

using namespace Legion;

/*static*/ int64_t DensifyTask::gpu_variant(const Task *task,
                                            const std::vector<PhysicalRegion> &regions,
                                            Context context,
                                            Runtime *runtime)
{
  Deserializer ctx{task, regions};

  std::vector<Column<true>> inputs;
  std::vector<OutputColumn> outputs;

  deserialize(ctx, inputs);
  deserialize(ctx, outputs);

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  std::vector<cudf::column_view> columns;
  for (auto &input : inputs) columns.push_back(std::move(to_cudf_column(input, stream)));

  cudf::table_view input_table(std::move(columns));

  DeferredBufferAllocator mr;

  auto result = std::make_unique<cudf::table>(input_table, stream, &mr);
  from_cudf_table(outputs, std::move(result), stream, mr);

  return static_cast<int64_t>(inputs[0].num_elements());
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

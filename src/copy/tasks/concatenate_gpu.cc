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

#include "copy/tasks/concatenate.h"
#include "cudf_util/allocators.h"
#include "cudf_util/column.h"
#include "util/gpu_task_context.h"

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/detail/concatenate.hpp>

namespace legate {
namespace pandas {
namespace copy {

using namespace Legion;

/*static*/ int64_t ConcatenateTask::gpu_variant(const Task *task,
                                                const std::vector<PhysicalRegion> &regions,
                                                Context context,
                                                Runtime *runtime)
{
  Deserializer ctx{task, regions};

  ConcatenateArgs args;
  deserialize(ctx, args);

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  std::vector<cudf::table_view> input_tables;
  for (auto &input_table : args.input_tables) {
    input_tables.push_back(std::move(to_cudf_table(input_table, stream)));
  }

  DeferredBufferAllocator mr;
  auto result      = cudf::detail::concatenate(input_tables, stream, &mr);
  auto result_size = static_cast<int64_t>(result->num_rows());

  from_cudf_table(args.output_table, std::move(result), stream, mr);

  return result_size;
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

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

#include "copy/tasks/drop_duplicates.h"
#include "column/column.h"
#include "cudf_util/allocators.h"
#include "cudf_util/column.h"
#include "cudf_util/types.h"
#include "util/cuda_helper.h"
#include "util/gpu_task_context.h"
#include "deserializer.h"

#include <cudf/copying.hpp>
#include <cudf/detail/concatenate.cuh>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>

namespace legate {
namespace pandas {
namespace copy {

using namespace Legion;

/*static*/ int64_t DropDuplicatesTask::gpu_variant(const Task *task,
                                                   const std::vector<PhysicalRegion> &regions,
                                                   Context context,
                                                   Runtime *runtime)
{
  Deserializer ctx{task, regions};

  KeepMethod method;
  deserialize(ctx, method);

  std::vector<int32_t> subset;
  deserialize(ctx, subset);

  uint32_t num_inputs{0};
  deserialize(ctx, num_inputs);

  std::vector<std::vector<Column<true>>> input_columns;
  std::vector<OutputColumn> outputs;

  for (auto idx = 0; idx < num_inputs; ++idx) {
    std::vector<Column<true>> columns;
    deserialize(ctx, columns);
    if (!columns[0].valid()) continue;
    input_columns.push_back(std::move(columns));
  }

  deserialize(ctx, outputs);

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  std::unique_ptr<cudf::table> concatenated;
  cudf::table_view to_dedup;

  if (input_columns.size() > 1) {
    std::vector<cudf::table_view> tables;
    for (auto &columns : input_columns) tables.push_back(to_cudf_table(columns, stream));
    concatenated = cudf::detail::concatenate(tables, stream);
    to_dedup     = concatenated->view();
  } else
    to_dedup = to_cudf_table(input_columns[0], stream);

  DeferredBufferAllocator mr;
  auto result = cudf::detail::drop_duplicates(
    to_dedup, subset, to_cudf_keep_option(method), cudf::null_equality::EQUAL, stream, &mr);
  auto result_size = static_cast<int64_t>(result->num_rows());
  from_cudf_table(outputs, std::move(result), stream, mr);
  return result_size;
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

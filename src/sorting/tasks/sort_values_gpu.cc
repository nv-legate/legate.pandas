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

#include <numeric>

#include "sorting/tasks/sort_values.h"
#include "column/device_column.h"
#include "util/gpu_task_context.h"
#include "util/zip_for_each.h"

#include <cudf/detail/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

namespace legate {
namespace pandas {
namespace sorting {

using namespace Legion;

using CudfColumns = std::vector<cudf::column_view>;

/*static*/ int64_t SortValuesTask::gpu_variant(const Task *task,
                                               const std::vector<PhysicalRegion> &regions,
                                               Context context,
                                               Runtime *runtime)
{
  Deserializer ctx{task, regions};

  SortValuesArgs args;
  deserialize(ctx, args);

  int64_t size = static_cast<int64_t>(args.input[0].num_elements());

  if (size == 0) {
    for (auto &column : args.output) column.make_empty();
    return 0;
  }

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  CudfColumns keys, values;
  for (auto &column : args.input)
    values.push_back(DeviceColumn<true>{column}.to_cudf_column(stream));
  for (auto idx : args.key_indices) keys.push_back(values[idx]);

  std::vector<cudf::order> column_order;
  std::vector<cudf::null_order> null_precedence;
  for (auto asc : args.ascending) {
    column_order.push_back(asc ? cudf::order::ASCENDING : cudf::order::DESCENDING);
    null_precedence.push_back(asc == args.put_null_first ? cudf::null_order::BEFORE
                                                         : cudf::null_order::AFTER);
  }

  DeferredBufferAllocator mr;
  auto result = cudf::detail::sort_by_key(cudf::table_view{std::move(values)},
                                          cudf::table_view{std::move(keys)},
                                          column_order,
                                          null_precedence,
                                          stream,
                                          &mr);

  util::for_each(result->view(), args.output, [&](auto &cudf_output, auto &output) {
    DeviceOutputColumn{output}.return_from_cudf_column(mr, cudf_output, stream);
  });

  return size;
}

}  // namespace sorting
}  // namespace pandas
}  // namespace legate

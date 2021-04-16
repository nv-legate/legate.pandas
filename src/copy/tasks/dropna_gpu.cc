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

#include "copy/tasks/dropna.h"
#include "copy/materialize.cuh"
#include "cudf_util/allocators.h"
#include "cudf_util/column.h"
#include "util/gpu_task_context.h"
#include "util/zip_for_each.h"

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/detail/stream_compaction.hpp>

#include <thrust/device_vector.h>

namespace legate {
namespace pandas {
namespace copy {

using namespace Legion;

using DropNaArg = DropNaTask::DropNaTaskArgs::DropNaArg;

/*static*/ int64_t DropNaTask::gpu_variant(const Task *task,
                                           const std::vector<PhysicalRegion> &regions,
                                           Context context,
                                           Runtime *runtime)
{
  Deserializer ctx{task, regions};

  DropNaTaskArgs args;
  deserialize(ctx, args);

  const Rect<1> in_rect = args.pairs[0].second.shape();

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  DeferredBufferAllocator mr;

  std::vector<cudf::column_view> input_columns;
  std::unique_ptr<cudf::column> materialized{nullptr};

  for (auto &pair : args.pairs) {
    if (pair.second.valid())
      input_columns.push_back(to_cudf_column(pair.second, stream));
    else {
      materialized =
        materialize(in_rect, args.range_start.value(), args.range_step.value(), stream, &mr);
      input_columns.push_back(materialized->view());
    }
  }

  cudf::table_view input_table{std::move(input_columns)};
  auto cudf_output =
    cudf::detail::drop_nulls(input_table, args.key_indices, args.keep_threshold, stream, &mr);
  auto output_size = static_cast<int64_t>(cudf_output->num_rows());

  auto cudf_outputs = cudf_output->release();
  util::for_each(args.pairs, cudf_outputs, [&](auto &pair, auto &cudf_output) {
    from_cudf_column(pair.first, std::move(cudf_output), stream, mr);
  });

  return output_size;
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

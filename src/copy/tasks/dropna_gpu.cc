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
#include "column/device_column.h"
#include "cudf_util/allocators.h"
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

using DropNaArg   = DropNaTask::DropNaTaskArgs::DropNaArg;
using CudfColumns = std::vector<cudf::column_view>;

static inline OutputColumn &output(DropNaArg &arg) { return arg.first; };

static inline Column<true> &input(DropNaArg &arg) { return arg.second; };

/*static*/ int64_t DropNaTask::gpu_variant(const Task *task,
                                           const std::vector<PhysicalRegion> &regions,
                                           Context context,
                                           Runtime *runtime)
{
  Deserializer ctx{task, regions};

  DropNaTaskArgs args;
  deserialize(ctx, args);

  const Rect<1> in_rect = input(args.pairs[0]).shape();

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  DeferredBufferAllocator mr;

  CudfColumns input_columns;
  std::vector<DeviceOutputColumn> output_columns;
  output_columns.reserve(args.pairs.size());
  std::unique_ptr<cudf::column> materialized{nullptr};

  for (auto &pair : args.pairs) {
    auto in = input(pair);
    DeviceOutputColumn out{output(pair)};

    if (in.valid())
      input_columns.push_back(DeviceColumn<true>{in}.to_cudf_column(stream));
    else {
      materialized =
        materialize(in_rect, args.range_start.value(), args.range_step.value(), stream, &mr);
      input_columns.push_back(materialized->view());
    }
    output_columns.push_back(out);
  }

  cudf::table_view input_table{std::move(input_columns)};
  auto cudf_output =
    cudf::detail::drop_nulls(input_table, args.key_indices, args.keep_threshold, stream, &mr);
  auto cudf_output_view = cudf_output->view();

  util::for_each(output_columns, cudf_output_view, [&](auto &output, auto &cudf_output) {
    output.return_from_cudf_column(mr, cudf_output, stream);
  });
  return static_cast<int64_t>(cudf_output->num_rows());
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

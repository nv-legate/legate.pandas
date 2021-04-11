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

#include "io/tasks/to_csv.h"
#include "io/file_util.h"
#include "column/device_column.h"
#include "util/cuda_helper.h"
#include "util/gpu_task_context.h"
#include "deserializer.h"

#include <cudf/io/data_sink.hpp>
#include <cudf/io/detail/csv.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>

namespace legate {
namespace pandas {
namespace io {
namespace csv {

using namespace Legion;

/*static*/ void ToCSVTask::gpu_variant(const Task *task,
                                       const std::vector<PhysicalRegion> &regions,
                                       Context context,
                                       Runtime *runtime)
{
  Deserializer ctx{task, regions};

  ToCSVArgs args;
  deserialize(ctx, args);

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  std::vector<cudf::column_view> columns;
  for (auto &column : args.columns)
    columns.push_back(DeviceColumn<true>(column).to_cudf_column(stream));

  auto task_id = static_cast<uint32_t>(task->index_point[0]);
  auto path    = args.partition
                ? get_partition_filename(std::move(args.path), ".", "", args.num_pieces, task_id)
                : args.path;

  auto sink = cudf::io::data_sink::create(path);
  cudf::io::table_metadata metadata;
  metadata.column_names = std::move(args.column_names);
  auto options          = cudf::io::csv_writer_options_builder()
                   .metadata(&metadata)
                   .na_rep(args.na_rep)
                   .include_header(args.header)
                   .rows_per_chunk(args.chunksize)
                   .line_terminator(args.line_terminator)
                   .inter_column_delimiter(args.sep[0])
                   .build();
  cudf::io::detail::csv::writer writer(
    std::move(sink), options, rmm::mr::get_current_device_resource());
  writer.write(cudf::table_view{std::move(columns)}, &metadata, stream);
  SYNC_AND_CHECK_STREAM(stream);
}

}  // namespace csv
}  // namespace io
}  // namespace pandas
}  // namespace legate

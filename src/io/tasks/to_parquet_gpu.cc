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

#include "io/tasks/to_parquet.h"
#include "io/file_util.h"
#include "cudf_util/column.h"
#include "cudf_util/types.h"
#include "util/cuda_helper.h"
#include "util/gpu_task_context.h"
#include "util/zip_for_each.h"
#include "deserializer.h"

#include <cudf/io/data_sink.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>

namespace legate {
namespace pandas {
namespace io {
namespace parquet {

using namespace Legion;

/*static*/ void ToParquetTask::gpu_variant(const Task *task,
                                           const std::vector<PhysicalRegion> &regions,
                                           Context context,
                                           Runtime *runtime)
{
  Deserializer ctx{task, regions};

  ToParquetArgs args;
  deserialize(ctx, args);

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  std::vector<cudf::column_view> columns;
  for (auto &column : args.columns) columns.push_back(to_cudf_column(column, stream));
  cudf::table_view table(std::move(columns));

  auto task_id = static_cast<uint32_t>(task->index_point[0]);
  auto path =
    get_partition_filename(std::move(args.path), "/", ".parquet", args.num_pieces, task_id);
  auto sink = cudf::io::data_sink::create(path);

  cudf::io::table_input_metadata metadata(table,
                                          {std::make_pair("pandas", std::move(args.metadata))});
  util::for_each(metadata.column_metadata,
                 args.column_names,
                 [&](auto &metadata, auto &column_name) { metadata.set_name(column_name); });

  auto options = cudf::io::parquet_writer_options_builder()
                   .metadata(&metadata)
                   .compression(to_cudf_compression(args.compression))
                   .build();
  cudf::io::detail::parquet::writer writer(std::move(sink),
                                           options,
                                           cudf::io::detail::SingleWriteMode::YES,
                                           rmm::mr::get_current_device_resource(),
                                           stream);
  writer.write(table);
  SYNC_AND_CHECK_STREAM(stream);
}

}  // namespace parquet
}  // namespace io
}  // namespace pandas
}  // namespace legate

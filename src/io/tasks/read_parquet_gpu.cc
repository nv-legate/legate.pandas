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

#include "io/tasks/read_parquet.h"
#include "column/device_column.h"
#include "cudf_util/allocators.h"
#include "util/gpu_task_context.h"
#include "util/zip_for_each.h"

#include <cudf/null_mask.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/detail/concatenate.hpp>

#include <arrow/api.h>
#include <arrow/filesystem/api.h>
#include <parquet/api/reader.h>

namespace legate {
namespace pandas {
namespace io {
namespace parquet {

using ReadParquetArgs = ReadParquetTask::ReadParquetArgs;

/*static*/ int64_t ReadParquetTask::gpu_variant(const Legion::Task *task,
                                                const std::vector<Legion::PhysicalRegion> &regions,
                                                Legion::Context context,
                                                Legion::Runtime *runtime)
{
  Deserializer ctx{task, regions};

  ReadParquetArgs args;
  deserialize(ctx, args);

  cudf::io::parquet_reader_options parquet_args;
  parquet_args.set_columns(args.column_names_to_parse);

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  DeferredBufferAllocator mr;

  auto return_columns = [&](auto result_view) {
    util::for_each(args.columns, result_view, [&](auto &output, auto &input) {
      DeviceOutputColumn{output}.return_from_cudf_column(mr, input, stream);
    });
  };

  int64_t task_id   = task->index_point[0];
  int64_t num_tasks = static_cast<int64_t>(task->index_domain.get_volume());

  int64_t num_rows = 0;
  if (args.filenames.size() == 1) {
    if (num_tasks > 1) {
      // Read metadata to figure out the number of rows
      auto *pool    = arrow::default_memory_pool();
      auto maybe_fs = arrow::fs::FileSystemFromUriOrPath("/");
#ifdef DEBUG_PANDAS
      assert(maybe_fs.ok());
#endif

      auto maybe_file = (*maybe_fs)->OpenInputFile(args.filenames.front());
#ifdef DEBUG_PANDAS
      assert(maybe_file.ok());
#endif

      auto arrow_reader = ::parquet::ParquetFileReader::Open(*maybe_file);
      auto metadata     = arrow_reader->metadata();

      int64_t total_num_rows = metadata->num_rows();
      int64_t skip_rows      = task_id * total_num_rows / num_tasks;
      int64_t num_rows_to_read =
        std::min((task_id + 1) * total_num_rows / num_tasks, total_num_rows) - skip_rows;

      if (num_rows_to_read <= 0)
        for (auto &column : args.columns) column.make_empty();
      else {
        num_rows = static_cast<int64_t>(num_rows_to_read);
        parquet_args.set_skip_rows(static_cast<cudf::size_type>(skip_rows));
        parquet_args.set_num_rows(static_cast<cudf::size_type>(num_rows));

        cudf::io::detail::parquet::reader reader{args.filenames, parquet_args, &mr};
        auto result = reader.read(parquet_args, stream);
        return_columns(result.tbl->view());
      }
    } else {
      cudf::io::detail::parquet::reader reader{args.filenames, parquet_args, &mr};
      auto result = reader.read(parquet_args, stream);
      return_columns(result.tbl->view());
      num_rows = static_cast<int64_t>(result.tbl->num_rows());
    }
  } else {
    int64_t num_files = args.filenames.size();
    int64_t start_idx = task_id * num_files / num_tasks;
    int64_t end_idx   = (task_id + 1) * num_files / num_tasks;

    std::vector<std::string> filenames;
    for (auto i = start_idx; i < end_idx; ++i) filenames.push_back(args.filenames[i]);

    if (filenames.empty()) {
      for (auto &column : args.columns) column.make_empty();
    } else {
      cudf::io::detail::parquet::reader reader{filenames, parquet_args, &mr};
      auto result = reader.read(parquet_args, stream);
      return_columns(result.tbl->view());
      num_rows = static_cast<int64_t>(result.tbl->num_rows());
    }
  }

  return num_rows;
}

}  // namespace parquet
}  // namespace io
}  // namespace pandas
}  // namespace legate

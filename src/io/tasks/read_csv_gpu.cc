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

#include <sstream>
#include <unordered_set>

#include <sys/stat.h>

#include "io/tasks/read_csv.h"
#include "column/device_column.h"
#include "cudf_util/allocators.h"
#include "cudf_util/types.h"
#include "util/gpu_task_context.h"
#include "util/zip_for_each.h"

#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/io/detail/csv.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/detail/concatenate.hpp>

namespace legate {
namespace pandas {
namespace io {
namespace csv {

using ReadCSVArgs = ReadCSVTask::ReadCSVArgs;

size_t find_size(const char *name)
{
  struct stat st;
  assert(stat(name, &st) == 0);
  return static_cast<size_t>(st.st_size);
}

inline const char *to_cudf_type_string(TypeCode type_code)
{
  switch (type_code) {
    case TypeCode::BOOL: {
      return ":bool";
    }
    case TypeCode::INT8: {
      return ":int8";
    }
    case TypeCode::INT16: {
      return ":int16";
    }
    case TypeCode::INT32: {
      return ":int32";
    }
    case TypeCode::INT64: {
      return ":int64";
    }
    case TypeCode::UINT8: {
      return ":uint8";
    }
    case TypeCode::UINT16: {
      return ":uint16";
    }
    case TypeCode::UINT32: {
      return ":uint32";
    }
    case TypeCode::UINT64: {
      return ":uint64";
    }
    case TypeCode::FLOAT: {
      return ":float32";
    }
    case TypeCode::DOUBLE: {
      return ":float64";
    }
    case TypeCode::STRING: {
      return ":str";
    }
    default: {
      assert(false);
      return nullptr;
    }
  }
  return nullptr;
}

bool can_split_within_file(const ReadCSVArgs &args)
{
  for (auto compression : args.compressions)
    if (compression != CompressionType::UNCOMPRESSED) return false;
  return args.skiprows == 0 && args.skipfooter == 0 && !args.nrows.valid();
}

/*static*/ int64_t ReadCSVTask::gpu_variant(const Legion::Task *task,
                                            const std::vector<Legion::PhysicalRegion> &regions,
                                            Legion::Context context,
                                            Legion::Runtime *runtime)
{
  Deserializer ctx{task, regions};

  ReadCSVArgs args;
  deserialize(ctx, args);

  std::unordered_set<int32_t> date_columns;
  for (auto const &date_column : args.date_columns) date_columns.insert(date_column);

  cudf::io::csv_reader_options csv_args;

  csv_args.set_header(-1);
  csv_args.set_delimiter(args.delimiter[0]);
  csv_args.set_skiprows(args.skiprows);
  csv_args.set_skipfooter(args.skipfooter);
  if (args.nrows.valid()) csv_args.set_nrows(*args.nrows);
  csv_args.set_quotechar(args.quotechar[0]);
  csv_args.enable_doublequote(args.doublequote);
  csv_args.enable_skip_blank_lines(args.skip_blank_lines);
  if (args.true_values.valid()) csv_args.set_true_values(std::move(*args.true_values));
  if (args.false_values.valid()) csv_args.set_false_values(std::move(*args.false_values));
  if (args.na_values.valid()) csv_args.set_na_values(std::move(*args.na_values));

  csv_args.set_infer_date_indexes(args.date_columns);

  std::vector<std::string> col_names;
  std::vector<std::string> dtypes;
  for (size_t idx = 0; idx < args.columns.size(); ++idx) {
    std::stringstream ss;
    ss << "f" << idx;
    col_names.push_back(ss.str());
    if (date_columns.find(idx) == date_columns.end())
      ss << to_cudf_type_string(args.columns[idx].code());
    else
      ss << ":timestamp[ns]";
    dtypes.push_back(ss.str());
  }

  csv_args.set_names(col_names);
  csv_args.set_dtypes(dtypes);

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
    csv_args.set_compression(to_cudf_compression(args.compressions[0]));
    if (can_split_within_file(args)) {
      int64_t total_size = find_size(args.filenames.begin()->c_str());
      int64_t offset     = total_size * task_id / num_tasks;
      int64_t num_bytes =
        std::max(0L, std::min(total_size * (task_id + 1) / num_tasks, total_size) - offset);

      if (num_bytes > 0) {
        csv_args.set_byte_range_offset(offset);
        csv_args.set_byte_range_size(num_bytes);
        cudf::io::detail::csv::reader reader{args.filenames, csv_args, &mr};

        auto result = reader.read(stream);
        return_columns(result.tbl->view());
        num_rows = static_cast<int64_t>(result.tbl->num_rows());
      } else {
        for (auto &column : args.columns) column.make_empty(true);
        num_rows = 0;
      }
    } else {
      cudf::io::detail::csv::reader reader{args.filenames, csv_args, &mr};
      auto result            = reader.read(stream);
      int64_t total_num_rows = result.tbl->num_rows();
      int64_t offset         = total_num_rows * task_id / num_tasks;
      num_rows =
        std::max(0L, std::min(total_num_rows, total_num_rows * (task_id + 1) / num_tasks) - offset);

      if (num_rows > 0) {
        auto sliced = cudf::slice(
          result.tbl->view(),
          {static_cast<cudf::size_type>(offset), static_cast<cudf::size_type>(offset + num_rows)});
        // Make a copy of the slice to return
        cudf::table copy(sliced, stream, &mr);
        return_columns(copy.view());
      } else {
        for (auto &column : args.columns) column.make_empty(true);
      }
    }
  } else {
    // Here we simply assume that the user provided an enough number of files to keep all GPUs busy
    int64_t num_files = args.filenames.size();
    int64_t start_idx = task_id * num_files / num_tasks;
    int64_t end_idx   = (task_id + 1) * num_files / num_tasks;

    std::vector<std::string> filenames;
    std::vector<CompressionType> compressions;
    for (auto i = start_idx; i < end_idx; ++i) {
      filenames.push_back(args.filenames[i]);
      compressions.push_back(args.compressions[i]);
    }

    if (filenames.empty()) {
      for (auto &column : args.columns) column.make_empty();
    } else {
      std::vector<std::unique_ptr<cudf::table>> tables;
      util::for_each(filenames, compressions, [&](auto &filename, auto &compression) {
        csv_args.set_compression(to_cudf_compression(compression));
        cudf::io::detail::csv::reader reader{{filename}, csv_args, &mr};
        tables.push_back(std::move(reader.read(stream).tbl));
      });

      std::vector<cudf::table_view> table_views;
      for (auto &table : tables) table_views.push_back(table->view());

      auto result = cudf::detail::concatenate(table_views, stream, &mr);
      return_columns(result->view());
      num_rows = static_cast<int64_t>(result->num_rows());
    }
  }

  return num_rows;
}

}  // namespace csv
}  // namespace io
}  // namespace pandas
}  // namespace legate

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

#include <ctime>
#include <sstream>
#include <unordered_set>

#include "io/tasks/read_parquet.h"

#include "deserializer.h"
#include "io/arrow_util.h"
#include "util/zip_for_each.h"

#include <arrow/api.h>
#include <arrow/table.h>
#include <arrow/filesystem/api.h>
#include <parquet/api/reader.h>
#include <parquet/arrow/reader.h>
#include <arrow/io/api.h>

namespace legate {
namespace pandas {
namespace io {
namespace parquet {

using ReadParquetArgs = ReadParquetTask::ReadParquetArgs;

namespace detail {

class ParquetReader : public Reader {
 public:
  ParquetReader(const ReadParquetArgs &args);

 public:
  virtual std::shared_ptr<arrow::Table> read(const std::string &filename) override;

 private:
  arrow::MemoryPool *pool;
  std::shared_ptr<arrow::fs::FileSystem> filesystem;
  std::vector<std::string> column_names;
  std::vector<int> column_indices;
};

ParquetReader::ParquetReader(const ReadParquetArgs &args) : column_names(args.column_names_to_parse)
{
  pool          = arrow::default_memory_pool();
  auto maybe_fs = arrow::fs::FileSystemFromUriOrPath("/");
#ifdef DEBUG_PANDAS
  assert(maybe_fs.ok());
#endif
  filesystem = *maybe_fs;
}

std::shared_ptr<arrow::Table> ParquetReader::read(const std::string &filename)
{
  auto maybe_file = filesystem->OpenInputFile(filename);
#ifdef DEBUG_PANDAS
  assert(maybe_file.ok());
#endif

  std::unique_ptr<::parquet::arrow::FileReader> reader;
  auto status = ::parquet::arrow::FileReader::Make(
    pool, ::parquet::ParquetFileReader::Open(*maybe_file), &reader);
#ifdef DEBUG_PANDAS
  assert(status.ok());
#endif

  if (!column_names.empty()) {
    if (column_indices.empty()) {
      std::shared_ptr<arrow::Schema> schema;
      status = reader->GetSchema(&schema);
#ifdef DEBUG_PANDAS
      assert(status.ok());
#endif
      for (auto &name : column_names) {
        auto column_index = schema->GetFieldIndex(name);
#ifdef DEBUG_PANDAS
        assert(column_index != -1);
#endif
        column_indices.push_back(column_index);
      }
    }
#ifdef DEBUG_PANDAS
    else {
      std::shared_ptr<arrow::Schema> schema;
      status = reader->GetSchema(&schema);
      assert(status.ok());

      for (auto idx = 0; idx < column_names.size(); ++idx)
        assert(column_indices[idx] == schema->GetFieldIndex(column_names[idx]));
    }
#endif
  }

  std::shared_ptr<arrow::Table> result;
  if (column_indices.empty())
    status = reader->ReadTable(&result);
  else
    status = reader->ReadTable(column_indices, &result);
#ifdef DEBUG_PANDAS
  assert(status.ok());
#endif
  return result;
}

}  // namespace detail

using namespace Legion;

/*static*/ int64_t ReadParquetTask::cpu_variant(const Task *task,
                                                const std::vector<PhysicalRegion> &regions,
                                                Context context,
                                                Runtime *runtime)
{
  Deserializer ctx{task, regions};

  ReadParquetArgs args;
  deserialize(ctx, args);

  auto task_id   = task->index_point[0];
  auto num_tasks = task->index_domain.get_volume();

  auto result =
    read_files(std::make_unique<detail::ParquetReader>(args), args.filenames, task_id, num_tasks);

  if (nullptr == result) {
    for (auto &column : args.columns) column.make_empty();
    return 0;
  }

  std::vector<std::shared_ptr<arrow::ChunkedArray>> in_columns;
  int64_t num_rows = 0;
  if (args.filenames.size() == 1)
    num_rows = slice_columns(in_columns, result, task_id, num_tasks);
  else {
    num_rows   = result->num_rows();
    in_columns = std::move(result->columns());
  }

  if (num_rows == 0)
    for (auto &column : args.columns) column.make_empty(true);
  else
    util::for_each(
      args.columns, in_columns, [&](auto &output, auto &input) { copy_column(output, input); });

  return num_rows;
}

void deserialize(Deserializer &ctx, ReadParquetArgs &args)
{
  uint32_t num_files = 0;
  deserialize(ctx, num_files);
#ifdef DEBUG_PANDAS
  assert(num_files >= 1);
#endif
  args.filenames.resize(num_files);
  for (auto &f : args.filenames) deserialize(ctx, f);

  uint32_t num_column_names = 0;
  deserialize(ctx, num_column_names);
  args.column_names_to_parse.resize(num_column_names);
  for (auto &name : args.column_names_to_parse) deserialize(ctx, name);

  uint32_t num_columns = 0;
  deserialize(ctx, num_columns);
  args.columns.resize(num_columns);
  for (auto &column : args.columns) deserialize(ctx, column);
}

static void __attribute__((constructor)) register_tasks(void)
{
  ReadParquetTask::register_variants_with_return<int64_t, int64_t>();
}

}  // namespace parquet
}  // namespace io
}  // namespace pandas
}  // namespace legate

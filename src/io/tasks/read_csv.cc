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

#include "io/tasks/read_csv.h"

#include "io/arrow_util.h"
#include "util/datetime.h"
#include "util/zip_for_each.h"

#include <arrow/api.h>
#include <arrow/table.h>
#include <arrow/csv/api.h>
#include <arrow/dataset/api.h>
#include <arrow/filesystem/api.h>
#include <arrow/io/api.h>

namespace legate {
namespace pandas {
namespace io {
namespace csv {

using ReadCSVArgs = ReadCSVTask::ReadCSVArgs;

namespace detail {

static std::shared_ptr<arrow::DataType> to_arrow_dtype(TypeCode type_code)
{
  switch (type_code) {
    case TypeCode::BOOL: {
      return arrow::boolean();
    }
    case TypeCode::INT8: {
      return arrow::int8();
    }
    case TypeCode::INT16: {
      return arrow::int16();
    }
    case TypeCode::INT32: {
      return arrow::int32();
    }
    case TypeCode::INT64: {
      return arrow::int64();
    }
    case TypeCode::UINT8: {
      return arrow::uint8();
    }
    case TypeCode::UINT16: {
      return arrow::uint16();
    }
    case TypeCode::UINT32: {
      return arrow::uint32();
    }
    case TypeCode::UINT64: {
      return arrow::uint64();
    }
    case TypeCode::FLOAT: {
      return arrow::float32();
    }
    case TypeCode::DOUBLE: {
      return arrow::float64();
    }
    case TypeCode::STRING: {
      return arrow::utf8();
    }
    default: {
      assert(false);
      return arrow::null();
    }
  }
  return arrow::null();
}

class CSVReader : public Reader {
 public:
  CSVReader(ReadCSVArgs &args);

 public:
  virtual std::shared_ptr<arrow::Table> read(const std::string &filename) override;

 private:
  int64_t skipfooter;
  arrow::MemoryPool *pool;
  arrow::csv::ReadOptions read_options;
  arrow::csv::ParseOptions parse_options;
  arrow::csv::ConvertOptions convert_options;
};

CSVReader::CSVReader(ReadCSVArgs &args) : skipfooter(args.skipfooter)
{
  pool = arrow::default_memory_pool();

  read_options    = arrow::csv::ReadOptions::Defaults();
  parse_options   = arrow::csv::ParseOptions::Defaults();
  convert_options = arrow::csv::ConvertOptions::Defaults();

  // Override the default options
  read_options.use_threads = false;
  read_options.skip_rows   = args.skiprows;
  for (size_t idx = 0; idx < args.columns.size(); ++idx) {
    std::stringstream ss;
    ss << "f" << idx;
    read_options.column_names.push_back(ss.str());
  }

  std::unordered_set<int32_t> date_columns;
  for (auto const &date_column : args.date_columns) date_columns.insert(date_column);

  parse_options.delimiter          = args.delimiter[0];
  parse_options.quote_char         = args.quotechar[0];
  parse_options.double_quote       = args.doublequote;
  parse_options.ignore_empty_lines = args.skip_blank_lines;

  convert_options.strings_can_be_null = true;
  if (args.true_values.valid()) convert_options.true_values = std::move(*args.true_values);
  if (args.false_values.valid()) convert_options.false_values = std::move(*args.false_values);
  if (args.na_values.valid()) convert_options.null_values = std::move(*args.na_values);

  size_t idx = 0;
  for (auto const &column : args.columns) {
    std::shared_ptr<arrow::DataType> dtype;
    if (date_columns.find(idx) == date_columns.end())
      dtype = to_arrow_dtype(column.code());
    else
      dtype = arrow::utf8();
    convert_options.column_types[read_options.column_names[idx++]] = dtype;
  }
}

std::shared_ptr<arrow::Table> CSVReader::read(const std::string &filename)
{
  auto maybe_input = arrow::io::ReadableFile::Open(filename, pool);
  if (!maybe_input.ok()) {
    std::cerr << maybe_input.status() << std::endl;
    exit(-1);
  }
  auto reader =
    arrow::csv::TableReader::Make(pool, *maybe_input, read_options, parse_options, convert_options);
  if (!reader.ok()) {
    std::cerr << reader.status() << std::endl;
    exit(-1);
  }
  auto maybe_table = (*reader)->Read();
  if (!maybe_table.ok()) {
    std::cerr << maybe_table.status() << std::endl;
    exit(-1);
  }
  auto table = *maybe_table;
  if (skipfooter > 0) {
    int64_t num_rows = std::max(0L, table->num_rows() - skipfooter);
    return table->Slice(0, num_rows);
  } else
    return table;
}

void parse_datetime(OutputColumn &out, std::shared_ptr<arrow::ChunkedArray> in)
{
  out.allocate(in->length());

  auto out_b = out.bitmask();
  out_b.clear();

  auto out_v = out.raw_column<int64_t>();

  size_t offset = 0;
  for (unsigned chunk_idx = 0; chunk_idx < in->num_chunks(); ++chunk_idx) {
    auto chunk      = in->chunk(chunk_idx);
    auto chunk_size = chunk->length();
    auto in_array   = chunk->data();

    auto in_offsets = in_array->GetValues<int32_t>(1);
    auto in_chars   = reinterpret_cast<const char *>(in_array->buffers[2]->data());
    for (size_t i = 0; i < chunk_size; ++i) {
      if (in_offsets[i] == in_offsets[i + 1]) {
        out_b.set(offset + i, false);
        continue;
      }
      // This returns a timestamp in milliseconds
      int64_t r = parseDateTimeFormat(in_chars, in_offsets[i], in_offsets[i + 1] - 1, false);
      out_v[i]  = r * 1000000;
      out_b.set(offset + i, true);
    }
    out_v += chunk_size;
    offset += chunk_size;
  }
}

}  // namespace detail

using namespace Legion;

/*static*/ int64_t ReadCSVTask::cpu_variant(const Task *task,
                                            const std::vector<PhysicalRegion> &regions,
                                            Context context,
                                            Runtime *runtime)
{
  Deserializer ctx{task, regions};

  ReadCSVArgs args;
  deserialize(ctx, args);

  auto num_tasks = task->index_domain.get_volume();
  auto task_id   = task->index_point[0];

  auto result =
    read_files(std::make_unique<detail::CSVReader>(args), args.filenames, task_id, num_tasks);

  if (nullptr == result) {
    for (auto &column : args.columns) column.make_empty();
    return 0;
  }

  // XXX: Unfortunately, Arrow's CSV readers have a limitation that they can't skip more than a
  //      a single block worth of rows, which can only be 2GB big at maximum. So, it is not
  //      possible to parallelize the loading by setting different skip_rows in different
  //      worker tasks. Here, we simply have each task read the whole table and copy the
  //      corresponding sub-table.

  std::vector<std::shared_ptr<arrow::ChunkedArray>> in_columns;
  int64_t num_rows = 0;
  if (args.filenames.size() == 1)
    num_rows = slice_columns(in_columns, result, task_id, num_tasks, std::move(args.nrows));
  else if (args.nrows.valid())
    num_rows = slice_columns(in_columns, result, 0, 1, std::move(args.nrows));
  else {
    num_rows   = result->num_rows();
    in_columns = std::move(result->columns());
  }

  if (num_rows == 0)
    for (auto &column : args.columns) column.make_empty(true);
  else
    util::for_each(args.columns, in_columns, [&](auto &output, auto &input) {
      if (output.code() != TypeCode::TS_NS)
        copy_column(output, input);
      else
        detail::parse_datetime(output, input);
    });

  return num_rows;
}

void deserialize(Deserializer &ctx, ReadCSVArgs &args)
{
  deserialize(ctx, args.filenames);
  deserialize(ctx, args.compressions);
  deserialize(ctx, args.delimiter);
  deserialize(ctx, args.skiprows);
  deserialize(ctx, args.skipfooter);
  deserialize(ctx, args.nrows);
  deserialize(ctx, args.quotechar);
  deserialize(ctx, args.doublequote);
  deserialize(ctx, args.skip_blank_lines);
  deserialize(ctx, args.true_values);
  deserialize(ctx, args.false_values);
  deserialize(ctx, args.na_values);
  deserialize(ctx, args.columns);
  deserialize(ctx, args.date_columns);
}

static void __attribute__((constructor)) register_tasks(void)
{
  ReadCSVTask::register_variants_with_return<int64_t, int64_t>();
}

}  // namespace csv
}  // namespace io
}  // namespace pandas
}  // namespace legate

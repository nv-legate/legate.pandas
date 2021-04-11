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

#include <fstream>

#include "io/tasks/to_csv.h"
#include "io/file_util.h"
#include "column/detail/column.h"
#include "string/converter.h"
#include "util/type_dispatch.h"

namespace legate {
namespace pandas {
namespace io {
namespace csv {

using namespace Legion;

using ToCSVArgs = ToCSVTask::ToCSVArgs;

void ToCSVTask::ToCSVArgs::cleanup(void)
{
  for (auto &column : columns) column.destroy();
}

using ColumnView = detail::Column;

namespace detail {

template <bool nullable>
struct Print {
  template <TypeCode CODE,
            std::enable_if_t<is_numeric_type<CODE>::value || CODE == TypeCode::STRING> * = nullptr>
  void operator()(std::ostream &stream,
                  const ColumnView &column,
                  size_t idx,
                  const std::string &na_rep)
  {
    using VAL = pandas_type_of<CODE>;
    string::detail::Converter<VAL> convert;
    if (nullable)
      stream << (column.bitmask().get(idx) ? convert(column.element<VAL>(idx)) : na_rep);
    else
      stream << convert(column.element<VAL>(idx));
  }

  template <TypeCode CODE,
            std::enable_if_t<!is_numeric_type<CODE>::value && CODE != TypeCode::STRING> * = nullptr>
  void operator()(std::ostream &stream,
                  const ColumnView &column,
                  size_t idx,
                  const std::string &na_rep)
  {
    assert(false);
  }
};

void print_column(std::ostream &stream,
                  const ColumnView &column,
                  size_t idx,
                  const std::string &na_rep)
{
  if (column.nullable())
    type_dispatch(column.code(), Print<true>{}, stream, column, idx, na_rep);
  else
    type_dispatch(column.code(), Print<false>{}, stream, column, idx, na_rep);
}

void print_column(std::ostream &stream,
                  const std::string &column,
                  size_t idx,
                  const std::string &na_rep)
{
  stream << column;
}

template <typename Column>
void print_line(std::ostream &stream,
                const std::vector<Column> &columns,
                size_t row_idx,
                const ToCSVArgs &args)
{
  auto num_columns = columns.size();
  for (auto col_idx = 0; col_idx < num_columns; ++col_idx) {
    print_column(stream, columns[col_idx], row_idx, args.na_rep);
    stream << (col_idx == num_columns - 1 ? args.line_terminator : args.sep);
  }
}

}  // namespace detail

/*static*/ void ToCSVTask::cpu_variant(const Task *task,
                                       const std::vector<PhysicalRegion> &regions,
                                       Context context,
                                       Runtime *runtime)
{
  Deserializer ctx{task, regions};

  ToCSVArgs args;
  deserialize(ctx, args);

  std::vector<ColumnView> columns;
  for (auto &column : args.columns) columns.push_back(column.view());

  auto task_id = static_cast<uint32_t>(task->index_point[0]);
  auto path    = args.partition
                ? get_partition_filename(std::move(args.path), ".", "", args.num_pieces, task_id)
                : args.path;

  std::ofstream ofs;
  ofs.open(path, std::ofstream::out | std::ofstream::trunc);

  auto num_columns = args.column_names.size();
  if (args.header) detail::print_line(ofs, args.column_names, 0, args);

  std::stringstream ss{};
  auto size = columns.front().size();
  for (auto row_idx = 0; row_idx < size; ++row_idx) {
    detail::print_line(ss, columns, row_idx, args);
    if ((row_idx + 1) % args.chunksize == 0 || size - 1 == row_idx) {
      ofs << ss.str();
      ss = std::stringstream{};
    }
  }

  ofs.close();
}

void deserialize(Deserializer &ctx, ToCSVArgs &args)
{
  deserialize(ctx, args.num_pieces);
  deserialize(ctx, args.chunksize);
  deserialize(ctx, args.partition);
  deserialize(ctx, args.header);

  deserialize(ctx, args.path);
  deserialize(ctx, args.sep);
  deserialize(ctx, args.na_rep);
  deserialize(ctx, args.line_terminator);

  uint32_t num_columns = 0;
  deserialize(ctx, num_columns);
  args.column_names.resize(num_columns);
  args.columns.resize(num_columns);
  deserialize(ctx, args.column_names, false);
  deserialize(ctx, args.columns, false);
}

static void __attribute__((constructor)) register_tasks(void) { ToCSVTask::register_variants(); }

}  // namespace csv
}  // namespace io
}  // namespace pandas
}  // namespace legate

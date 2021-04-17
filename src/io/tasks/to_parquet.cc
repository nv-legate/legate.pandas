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

#include <cmath>
#include <iomanip>
#include <sstream>

#include "io/tasks/to_parquet.h"
#include "io/arrow_util.h"
#include "io/file_util.h"
#include "column/detail/column.h"
#include "deserializer.h"

#include <arrow/table.h>
#include <arrow/io/api.h>
#include <arrow/util/key_value_metadata.h>
#include <parquet/arrow/writer.h>

namespace legate {
namespace pandas {
namespace io {
namespace parquet {

using namespace Legion;

using ToParquetArgs = ToParquetTask::ToParquetArgs;
using ColumnView    = detail::Column;
using TableView     = detail::Table;

/*static*/ void ToParquetTask::cpu_variant(const Task *task,
                                           const std::vector<PhysicalRegion> &regions,
                                           Context context,
                                           Runtime *runtime)
{
  Deserializer ctx{task, regions};

  ToParquetArgs args;
  deserialize(ctx, args);

  std::vector<ColumnView> columns;
  for (auto &column : args.columns) columns.push_back(column.view());

  alloc::DeferredBufferAllocator allocator{};
  auto table = to_arrow(TableView{std::move(columns)}, args.column_names, allocator);
  if (!args.metadata.empty())
    table = table->ReplaceSchemaMetadata(
      arrow::key_value_metadata({"pandas"}, {std::move(args.metadata)}));

  std::shared_ptr<arrow::io::FileOutputStream> outfile;
  auto task_id = static_cast<uint32_t>(task->index_point[0]);
  auto path =
    get_partition_filename(std::move(args.path), "/", ".parquet", args.num_pieces, task_id);
  PARQUET_ASSIGN_OR_THROW(outfile, arrow::io::FileOutputStream::Open(path));

  auto writer_properties = ::parquet::WriterProperties::Builder()
                             .enable_dictionary()
                             ->compression(to_arrow_compression(args.compression))
                             ->build();
  auto arrow_properties = ::parquet::ArrowWriterProperties::Builder().store_schema()->build();
  PARQUET_THROW_NOT_OK(::parquet::arrow::WriteTable(*table,
                                                    arrow::default_memory_pool(),
                                                    outfile,
                                                    columns[0].size(),
                                                    writer_properties,
                                                    arrow_properties));
}

void deserialize(Deserializer &ctx, ToParquetArgs &args)
{
  deserialize(ctx, args.err_mkdir);
  deserialize(ctx, args.num_pieces);

  uint32_t compression{0};
  deserialize(ctx, compression);
  args.compression = static_cast<CompressionType>(compression);

  deserialize(ctx, args.path);
  deserialize(ctx, args.metadata);

  uint32_t num_columns = 0;
  deserialize(ctx, num_columns);
  args.column_names.resize(num_columns);
  args.columns.resize(num_columns);
  deserialize(ctx, args.column_names, false);
  deserialize(ctx, args.columns, false);
#ifdef DEBUG_PANDAS
  assert(args.err_mkdir == 0);
#endif
}

static void __attribute__((constructor)) register_tasks(void)
{
  ToParquetTask::register_variants();
}

}  // namespace parquet
}  // namespace io
}  // namespace pandas
}  // namespace legate

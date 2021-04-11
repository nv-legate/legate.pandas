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

#pragma once

#include "pandas.h"

#include "column/column.h"
#include "column/detail/column.h"

#include <arrow/util/compression.h>

namespace arrow {
class ChunkedArray;
class MemoryPool;
class Table;
}  // namespace arrow

namespace legate {
namespace pandas {
namespace io {

struct Reader {
  virtual std::shared_ptr<arrow::Table> read(const std::string &filename) = 0;
};

arrow::Compression::type to_arrow_compression(CompressionType compression);

std::shared_ptr<arrow::Table> read_files(std::unique_ptr<Reader> reader,
                                         const std::vector<std::string> &filenames,
                                         int64_t task_id,
                                         size_t num_tasks);

int64_t slice_columns(std::vector<std::shared_ptr<arrow::ChunkedArray>> &columns,
                      std::shared_ptr<arrow::Table> table,
                      int64_t task_id,
                      size_t num_tasks,
                      Maybe<int32_t> &&opt_nrows = Maybe<int32_t>{});

void copy_bitmask(Bitmask &out_b, std::shared_ptr<arrow::ChunkedArray> in);

void copy_column(OutputColumn &out, std::shared_ptr<arrow::ChunkedArray> in);

std::shared_ptr<arrow::Table> to_arrow(const detail::Table &table,
                                       const std::vector<std::string> &column_names,
                                       alloc::Allocator &allocator);

}  // namespace io
}  // namespace pandas
}  // namespace legate

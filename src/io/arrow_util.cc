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

#include "io/arrow_util.h"
#include "bitmask/compact_bitmask.h"
#include "util/allocator.h"
#include "util/type_dispatch.h"
#include "util/zip_for_each.h"

#include <arrow/api.h>
#include <arrow/compute/cast.h>

namespace legate {
namespace pandas {
namespace io {

using ColumnView = detail::Column;
using TableView  = detail::Table;

arrow::Compression::type to_arrow_compression(CompressionType compression)
{
  switch (compression) {
    case CompressionType::UNCOMPRESSED: return arrow::Compression::type::UNCOMPRESSED;
    case CompressionType::SNAPPY: return arrow::Compression::type::SNAPPY;
    case CompressionType::GZIP: return arrow::Compression::type::GZIP;
    case CompressionType::BROTLI: return arrow::Compression::type::BROTLI;
    case CompressionType::BZ2: {
      fprintf(stderr, "Unsupported compression kind: BZ2");
      break;
    }
    case CompressionType::ZIP: {
      fprintf(stderr, "Unsupported compression kind: ZIP");
      break;
    }
    case CompressionType::XZ: {
      fprintf(stderr, "Unsupported compression kind: XZ");
      break;
    }
  }
  assert(false);
}

std::shared_ptr<arrow::Table> read_files(std::unique_ptr<Reader> reader,
                                         const std::vector<std::string> &filenames,
                                         int64_t task_id,
                                         size_t num_tasks)
{
  std::vector<std::string> my_filenames;
  if (filenames.size() == 1)
    my_filenames.push_back(filenames.front());
  else {
    auto start_idx = filenames.size() * task_id / num_tasks;
    auto end_idx =
      std::min<int64_t>(filenames.size() * (task_id + 1) / num_tasks, filenames.size());
    for (auto i = start_idx; i < end_idx; ++i) my_filenames.push_back(filenames[i]);
  }

  std::vector<std::shared_ptr<arrow::Table>> tables;

  for (auto &filename : my_filenames) tables.push_back(reader->read(filename));

  if (tables.empty())
    return nullptr;
  else if (tables.size() == 1)
    return *tables.begin();
  else {
    auto *pool = arrow::default_memory_pool();
    auto maybe_table =
      arrow::ConcatenateTables(tables, arrow::ConcatenateTablesOptions::Defaults(), pool);
#if DEBUG_PANDAS
    assert(maybe_table.ok());
#endif
    return *maybe_table;
  }
}

int64_t slice_columns(std::vector<std::shared_ptr<arrow::ChunkedArray>> &columns,
                      std::shared_ptr<arrow::Table> table,
                      int64_t task_id,
                      size_t num_tasks,
                      Maybe<int32_t> &&opt_nrows)
{
  int64_t total_num_rows = static_cast<int64_t>(opt_nrows.valid() ? *opt_nrows : table->num_rows());
  int64_t num_rows_per_task = (total_num_rows + num_tasks - 1) / num_tasks;
  int64_t offset            = num_rows_per_task * task_id;
  int64_t num_rows          = std::min(num_rows_per_task, total_num_rows - offset);

  if (num_rows <= 0) return 0;

  for (auto &column : table->columns()) {
    // We need to uncompress boolean columns before copying them
    if (column->type() == arrow::boolean()) {
      auto converted = (*arrow::compute::Cast(column, arrow::uint8())).chunked_array();
      columns.push_back(converted->Slice(offset, num_rows));
    } else
      columns.push_back(column->Slice(offset, num_rows));
  }
  return num_rows;
}

void copy_bitmask(Bitmask &out_b, std::shared_ptr<arrow::ChunkedArray> in)
{
  auto out_offset = 0;
  for (unsigned chunk_idx = 0; chunk_idx < in->num_chunks(); ++chunk_idx) {
    auto chunk      = in->chunk(chunk_idx);
    auto chunk_size = chunk->length();
    auto in_array   = chunk->data();
    auto in_offset  = in_array->offset;

    if (in_array->buffers[0] == nullptr)
      for (unsigned i = 0; i < chunk_size; ++i) out_b.set(out_offset++);
    else {
      CompactBitmask in_b{in_array->buffers[0]->data(), static_cast<size_t>(chunk_size)};
      for (unsigned i = 0; i < chunk_size; ++i) out_b.set(out_offset++, in_b.get(in_offset++));
    }
  }
}

static void copy_string_column(OutputColumn &out, std::shared_ptr<arrow::ChunkedArray> in)
{
  // Compute size of the character container
  size_t num_chars = 0;
  std::vector<size_t> chars_sizes;
  for (unsigned chunk_idx = 0; chunk_idx < in->num_chunks(); ++chunk_idx) {
    auto chunk      = in->chunk(chunk_idx);
    auto chunk_size = chunk->length();
    auto in_array   = chunk->data();

    auto in_offsets = in_array->GetValues<int32_t>(1);
    chars_sizes.push_back(in_offsets[chunk_size] - in_offsets[0]);
    num_chars += chars_sizes.back();
  }

  // Allocate character container
  auto &chars_column = out.child(1);
  if (num_chars > 0) {
    chars_column.allocate(num_chars);
    auto *out_c = chars_column.raw_column<int8_t>();

    // Copy characters
    for (unsigned chunk_idx = 0; chunk_idx < in->num_chunks(); ++chunk_idx) {
      auto in_array   = in->chunk(chunk_idx)->data();
      auto in_chars   = in_array->buffers[2]->data();
      auto in_offsets = in_array->GetValues<int32_t>(1);
      auto chars_size = chars_sizes[chunk_idx];
      memcpy(out_c, &in_chars[in_offsets[0]], chars_size);
      out_c += chars_size;
    }
  } else
    chars_column.make_empty();

  // Convert offsets to ranges
  auto &offsets_column = out.child(0);
  if (in->length() == 0) {
    offsets_column.make_empty();
    return;
  }

  offsets_column.allocate(in->length() > 0 ? in->length() + 1 : 0);

  auto out_v          = offsets_column.raw_column<int32_t>();
  auto curr           = out_v;
  int32_t last_offset = 0;
  for (unsigned chunk_idx = 0; chunk_idx < in->num_chunks(); ++chunk_idx) {
    auto chunk      = in->chunk(chunk_idx);
    auto chunk_size = chunk->length();
    auto in_array   = chunk->data();

    if (chunk_idx == in->num_chunks() - 1) chunk_size += 1;

    auto in_offsets = in_array->GetValues<int32_t>(1);

    for (unsigned i = 0; i < chunk_size; ++i) curr[i] = in_offsets[i] + last_offset;
    curr += chunk_size;
    last_offset += in_offsets[chunk_size];
  }
  int32_t first_offset = out_v[0];
  for (; out_v != curr; ++out_v) *out_v -= first_offset;
}

namespace detail {

struct CopyChunk {
  template <TypeCode CODE>
  int8_t *operator()(int8_t *out, std::shared_ptr<arrow::Array> &&chunk)
  {
    using VAL       = pandas_type_of<CODE>;
    auto chunk_size = chunk->length() * sizeof(VAL);
    memcpy(out, chunk->data()->GetValues<VAL>(1), chunk_size);
    return out + chunk_size;
  }
};

}  // namespace detail

void copy_column(OutputColumn &out, std::shared_ptr<arrow::ChunkedArray> in)
{
  out.allocate(in->length());

  auto out_b = out.bitmask();
  copy_bitmask(out_b, in);

  if (out.code() == TypeCode::STRING) {
    copy_string_column(out, in);
    return;
  }

  auto out_v = static_cast<int8_t *>(out.raw_column_untyped());
  for (unsigned chunk_idx = 0; chunk_idx < in->num_chunks(); ++chunk_idx)
    out_v =
      type_dispatch_numeric_only(out.code(), detail::CopyChunk{}, out_v, in->chunk(chunk_idx));
}

namespace detail {

template <typename... Ts>
std::shared_ptr<arrow::Array> construct_arrow_array(TypeCode code, Ts &&... args)
{
  switch (code) {
    case TypeCode::BOOL: return std::make_shared<arrow::BooleanArray>(std::forward<Ts>(args)...);
    case TypeCode::INT8: return std::make_shared<arrow::Int8Array>(std::forward<Ts>(args)...);
    case TypeCode::INT16: return std::make_shared<arrow::Int16Array>(std::forward<Ts>(args)...);
    case TypeCode::INT32: return std::make_shared<arrow::Int32Array>(std::forward<Ts>(args)...);
    case TypeCode::INT64: return std::make_shared<arrow::Int64Array>(std::forward<Ts>(args)...);
    case TypeCode::UINT8: return std::make_shared<arrow::UInt8Array>(std::forward<Ts>(args)...);
    case TypeCode::UINT16: return std::make_shared<arrow::UInt16Array>(std::forward<Ts>(args)...);
    case TypeCode::UINT32: return std::make_shared<arrow::UInt32Array>(std::forward<Ts>(args)...);
    case TypeCode::UINT64: return std::make_shared<arrow::UInt64Array>(std::forward<Ts>(args)...);
    case TypeCode::FLOAT: return std::make_shared<arrow::FloatArray>(std::forward<Ts>(args)...);
    case TypeCode::DOUBLE: return std::make_shared<arrow::DoubleArray>(std::forward<Ts>(args)...);
    case TypeCode::TS_NS:
      return std::make_shared<arrow::TimestampArray>(arrow::timestamp(arrow::TimeUnit::NANO),
                                                     std::forward<Ts>(args)...);
    default: {
      assert(false);
      return nullptr;
    }
  }
}

template <typename T>
std::shared_ptr<arrow::Buffer> to_buffer(const ColumnView &column)
{
  if (column.size() == 0)
    return nullptr;
  else
    return arrow::Buffer::Wrap(column.column<T>(), column.size());
}

std::pair<std::shared_ptr<arrow::Buffer>, int64_t> to_bitmask_buffer(const ColumnView &column,
                                                                     alloc::Allocator &allocator)
{
  std::shared_ptr<arrow::Buffer> bitmask_buffer = nullptr;
  int64_t null_count{0};
  if (column.nullable()) {
    auto bitmask = column.bitmask();
    null_count   = bitmask.count_unset_bits();
    if (null_count > 0) {
      auto size = column.size();
      CompactBitmask compact_bitmask(size, allocator);
      for (auto idx = 0; idx < size; ++idx) compact_bitmask.set(idx, bitmask.get(idx));
      bitmask_buffer = std::make_shared<arrow::Buffer>(compact_bitmask.raw_ptr(), size);
    }
  }
  return std::make_pair(bitmask_buffer, null_count);
}

struct ToArrow {
  template <TypeCode CODE, std::enable_if_t<is_primitive_type<CODE>::value> * = nullptr>
  std::shared_ptr<arrow::Array> operator()(const ColumnView &column, alloc::Allocator &allocator)
  {
    auto bitmask = to_bitmask_buffer(column, allocator);
    return construct_arrow_array(column.code(),
                                 column.size(),
                                 to_buffer<pandas_type_of<CODE>>(column),
                                 bitmask.first,
                                 bitmask.second);
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::STRING> * = nullptr>
  std::shared_ptr<arrow::Array> operator()(const ColumnView &column, alloc::Allocator &allocator)
  {
    auto bitmask = to_bitmask_buffer(column, allocator);
    return std::make_shared<arrow::StringArray>(column.size(),
                                                to_buffer<int32_t>(column.child(0)),
                                                to_buffer<int8_t>(column.child(1)),
                                                bitmask.first,
                                                bitmask.second);
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::CAT32> * = nullptr>
  std::shared_ptr<arrow::Array> operator()(const ColumnView &column, alloc::Allocator &allocator)
  {
    assert(false);
    return nullptr;
  }
};

}  // namespace detail

std::shared_ptr<arrow::Table> to_arrow(const TableView &table,
                                       const std::vector<std::string> &column_names,
                                       alloc::Allocator &allocator)
{
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  std::vector<std::shared_ptr<arrow::Field>> fields;

  std::transform(table.columns().begin(),
                 table.columns().end(),
                 std::back_inserter(arrays),
                 [&](auto const &column) {
                   return type_dispatch(column.code(), detail::ToArrow{}, column, allocator);
                 });

  std::transform(
    arrays.begin(),
    arrays.end(),
    column_names.begin(),
    std::back_inserter(fields),
    [](auto const &array, auto const &name) { return arrow::field(name, array->type()); });

  return arrow::Table::Make(arrow::schema(fields), arrays);
}

}  // namespace io
}  // namespace pandas
}  // namespace legate

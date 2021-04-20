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

#include "cudf_util/column.h"
#include "util/zip_for_each.h"

#include <cudf/table/table.hpp>

namespace legate {
namespace pandas {

using namespace Legion;

void from_cudf_column(OutputColumn &column,
                      std::unique_ptr<cudf::column> &&cudf_column,
                      cudaStream_t stream,
                      DeferredBufferAllocator &allocator)
{
#ifdef DEBUG_PANDAS
  assert(to_cudf_type_id(column.code()) == cudf_column->type().id());
#endif
  auto num_elements = cudf_column->size();

  if (num_elements == 0) {
    column.make_empty();
    return;
  }

  auto contents = cudf_column->release();

  auto data = contents.data->data();
  if (nullptr != data) {
    auto column_buffer = allocator.pop_allocation(data);
    column.return_column_from_instance(column_buffer.get_instance(), num_elements);
  } else
    column.allocate_column(num_elements);

  if (column.nullable()) {
    column.allocate_bitmask(num_elements);
    if (nullptr != contents.null_mask->data()) {
      util::to_boolmask(column.raw_bitmask(),
                        reinterpret_cast<const Bitmask::AllocType *>(contents.null_mask->data()),
                        num_elements,
                        stream);
#ifdef DEBUG_PANDAS
      SYNC_AND_CHECK_STREAM(stream);
#endif
    } else {
      auto target = column.bitmask();
      target.set_all_valid(stream);
    }
  }

  for (auto idx = 0; idx < column.num_children() && idx < contents.children.size(); ++idx)
    from_cudf_column(column.child(idx), std::move(contents.children[idx]), stream, allocator);
}

cudf::table_view to_cudf_table(const std::vector<Column<true>> &columns, cudaStream_t stream)
{
  std::vector<cudf::column_view> column_views;
  for (auto &column : columns) column_views.push_back(to_cudf_column(column, stream));
  return cudf::table_view(std::move(column_views));
}

void from_cudf_table(std::vector<OutputColumn> &columns,
                     std::unique_ptr<cudf::table> &&cudf_table,
                     cudaStream_t stream,
                     DeferredBufferAllocator &allocator)
{
  auto cudf_columns = cudf_table->release();
  util::for_each(columns, cudf_columns, [&](auto &column, auto &cudf_column) {
    from_cudf_column(column, std::move(cudf_column), stream, allocator);
  });
}

}  // namespace pandas
}  // namespace legate

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

#include "column/column.h"
#include "cudf_util/allocators.h"
#include "cudf_util/bitmask.h"
#include "cudf_util/detail.h"
#include "cudf_util/types.h"
#include "util/cuda_helper.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

namespace legate {
namespace pandas {

template <bool READ>
const cudf::bitmask_type *to_cudf_bitmask(const Column<READ> &column, cudaStream_t stream)
{
  const auto num_bits = column.num_elements();

  auto boolmask = column.raw_bitmask_read();
  auto bitmask  = static_cast<Bitmask::AllocType *>(
    rmm::mr::get_current_device_resource()->allocate((num_bits + 63) / 64 * 64, stream));
  util::to_bitmask(bitmask, boolmask, num_bits, stream);
#ifdef DEBUG_PANDAS
  SYNC_AND_CHECK_STREAM(stream);
#endif
  return reinterpret_cast<const cudf::bitmask_type *>(bitmask);
}

template <bool READ>
cudf::column_view to_cudf_column(const Column<READ> &column, cudaStream_t stream)
{
  const auto p       = column.is_meta() ? nullptr : column.raw_column_untyped_read();
  const auto type_id = to_cudf_type_id(column.code());
  const auto size    = static_cast<cudf::size_type>(column.num_elements());

  auto null_mask  = static_cast<const cudf::bitmask_type *>(nullptr);
  auto null_count = static_cast<cudf::size_type>(0);

  if (size > 0 && column.nullable()) {
    null_mask  = to_cudf_bitmask(column, stream);
    null_count = cudf::detail::count_unset_bits(null_mask, 0, size, stream);
#ifdef DEBUG_PANDAS
    SYNC_AND_CHECK_STREAM(stream);
#endif
  }

  std::vector<cudf::column_view> children;
  for (auto child_idx = 0; child_idx < column.num_children(); ++child_idx)
    children.push_back(to_cudf_column(column.child(child_idx), stream));

  return cudf::column_view{cudf::data_type{type_id},
                           size,
                           p,
                           null_count == 0 ? nullptr : null_mask,
                           null_count,
                           0,
                           std::move(children)};
}

void from_cudf_column(OutputColumn &column,
                      std::unique_ptr<cudf::column> &&cudf_column,
                      cudaStream_t stream,
                      DeferredBufferAllocator &allocator);

cudf::table_view to_cudf_table(const std::vector<Column<true>> &columns, cudaStream_t stream);

void from_cudf_table(std::vector<OutputColumn> &columns,
                     std::unique_ptr<cudf::table> &&cudf_table,
                     cudaStream_t stream,
                     DeferredBufferAllocator &allocator);

}  // namespace pandas
}  // namespace legate

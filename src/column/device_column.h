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

#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/traits.hpp>

namespace legate {
namespace pandas {

template <bool READ>
class DeviceColumn {
 public:
  DeviceColumn(const Column<READ>& host_column) : host_column_(host_column) {}

 public:
  const cudf::bitmask_type* raw_bitmask_read(cudaStream_t stream) const;
  cudf::column_view to_cudf_column(cudaStream_t stream) const;

 private:
  const Column<READ>& host_column_;
};

class DeviceOutputColumn : public OutputColumn {
 public:
  DeviceOutputColumn()                                = default;
  DeviceOutputColumn(const DeviceOutputColumn& other) = default;
  DeviceOutputColumn(const OutputColumn& other);

 public:
  void destroy();

 public:
  void return_from_cudf_column(DeferredBufferAllocator& allocator,
                               cudf::column_view cudf_column,
                               cudaStream_t stream);
};

template <bool READ>
const cudf::bitmask_type* DeviceColumn<READ>::raw_bitmask_read(cudaStream_t stream) const
{
  const auto num_bits = host_column_.num_elements();

  auto boolmask = host_column_.raw_bitmask_read();
  auto bitmask  = static_cast<Bitmask::AllocType*>(
    rmm::mr::get_current_device_resource()->allocate((num_bits + 63) / 64 * 64, stream));
  util::to_bitmask(bitmask, boolmask, num_bits, stream);
  return reinterpret_cast<const cudf::bitmask_type*>(bitmask);
}

template <bool READ>
cudf::column_view DeviceColumn<READ>::to_cudf_column(cudaStream_t stream) const
{
  const auto p       = host_column_.is_meta() ? nullptr : host_column_.raw_column_untyped_read();
  const auto type_id = to_cudf_type_id(host_column_.code());
  const auto size    = static_cast<cudf::size_type>(host_column_.num_elements());

  auto null_mask  = static_cast<const cudf::bitmask_type*>(nullptr);
  auto null_count = static_cast<cudf::size_type>(0);

  if (size > 0 && host_column_.nullable()) {
    null_mask  = raw_bitmask_read(stream);
    null_count = cudf::detail::count_unset_bits(null_mask, 0, size, stream);
  }

  std::vector<cudf::column_view> children;
  for (auto child_idx = 0; child_idx < host_column_.num_children(); ++child_idx) {
    DeviceColumn<READ> child{host_column_.child(child_idx)};
    children.push_back(child.to_cudf_column(stream));
  }

  return cudf::column_view{cudf::data_type{type_id},
                           size,
                           p,
                           null_count == 0 ? nullptr : null_mask,
                           null_count,
                           0,
                           std::move(children)};
}

}  // namespace pandas
}  // namespace legate

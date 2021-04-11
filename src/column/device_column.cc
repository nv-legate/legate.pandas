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

#include "column/device_column.h"

namespace legate {
namespace pandas {

DeviceOutputColumn::DeviceOutputColumn(const OutputColumn& other) : OutputColumn(other) {}

void DeviceOutputColumn::destroy() { OutputColumn::destroy(); }

void DeviceOutputColumn::return_from_cudf_column(DeferredBufferAllocator& allocator,
                                                 cudf::column_view cudf_column,
                                                 cudaStream_t stream)
{
#ifdef DEBUG_PANDAS
  assert(to_cudf_type_id(code()) == cudf_column.type().id());
#endif
  auto num_elements = cudf_column.size();

  if (num_elements == 0) {
    make_empty();
    return;
  }

  num_elements_ = static_cast<size_t>(num_elements);

  auto data = cudf_column.head();
  if (nullptr != data) {
    auto column_buffer = allocator.pop_allocation(data);
    column_.return_from_instance(column_buffer.get_instance(), num_elements_, elem_size());
  } else
    column_.allocate(num_elements_);

  if (nullable()) {
    bitmask_->allocate(num_elements_);
    auto target = bitmask();
    if (cudf_column.nullable()) {
      CompactBitmask source{
        reinterpret_cast<const CompactBitmask::AllocType*>(cudf_column.null_mask()), num_elements_};
      source.to_boolmask(target, stream);
    } else
      target.set_all_valid(stream);
  }

  for (auto idx = 0; idx < OutputColumn::num_children() && idx < cudf_column.num_children(); ++idx)
    DeviceOutputColumn{OutputColumn::child(idx)}.return_from_cudf_column(
      allocator, cudf_column.child(idx), stream);
}

cudf::mutable_column_view DeviceOutputColumn::to_mutable_cudf_column() const
{
  auto p             = is_meta() ? nullptr : raw_column_untyped();
  const auto type_id = to_cudf_type_id(code());
  const auto size    = static_cast<cudf::size_type>(num_elements());

  std::vector<cudf::mutable_column_view> children;
  for (auto child_idx = 0; child_idx < num_children(); ++child_idx)
    children.push_back(DeviceOutputColumn{child(child_idx)}.to_mutable_cudf_column());

  return cudf::mutable_column_view{
    cudf::data_type{type_id}, size, p, nullptr, 0, 0, std::move(children)};
}

}  // namespace pandas
}  // namespace legate

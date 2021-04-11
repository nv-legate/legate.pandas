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

namespace legate {
namespace pandas {

////////////////////////////////////////
// Column
////////////////////////////////////////

template <bool READ>
inline void Column<READ>::destroy()
{
  column_.destroy();
  if (nullptr != bitmask_) { bitmask_->destroy(); }
}

template <bool READ>
template <typename T>
T *Column<READ>::raw_column_write() const
{
  return column_.template raw_write<T>();
}

template <bool READ>
template <typename T>
const T *Column<READ>::raw_column_read() const
{
  return column_.template raw_read<T>();
}

template <bool READ>
void *Column<READ>::raw_column_untyped_write() const
{
  return column_.raw_untyped_write();
}

template <bool READ>
const void *Column<READ>::raw_column_untyped_read() const
{
  return column_.raw_untyped_read();
}

template <bool READ>
template <typename T>
const AccessorWO<T, 1> &Column<READ>::write_accessor() const
{
  return column_.template write_accessor<T>();
}

template <bool READ>
template <typename T>
const AccessorRO<T, 1> &Column<READ>::read_accessor() const
{
  return column_.template read_accessor<T>();
}

template <bool READ>
Bitmask::AllocType *Column<READ>::raw_bitmask_write() const
{
  assert(nullptr != bitmask_);
  return bitmask_->template raw_write<Bitmask::AllocType>();
}

template <bool READ>
const Bitmask::AllocType *Column<READ>::raw_bitmask_read() const
{
  assert(nullptr != bitmask_);
  return bitmask_->template raw_read<Bitmask::AllocType>();
}

template <bool READ>
Bitmask Column<READ>::read_bitmask() const
{
  return Bitmask(raw_bitmask_read(), num_elements_);
}

template <bool READ>
Bitmask Column<READ>::write_bitmask() const
{
  return Bitmask(raw_bitmask_write(), num_elements_);
}

template <bool READ>
std::shared_ptr<Bitmask> Column<READ>::maybe_read_bitmask() const
{
  return nullptr == bitmask_ ? nullptr
                             : std::make_shared<Bitmask>(raw_bitmask_read(), num_elements_);
}

template <bool READ>
std::shared_ptr<Bitmask> Column<READ>::maybe_write_bitmask() const
{
  return nullptr == bitmask_ ? nullptr
                             : std::make_shared<Bitmask>(raw_bitmask_write(), num_elements_);
}

template <bool READ>
inline size_t Column<READ>::bitmask_bytes() const
{
  assert(nullptr != bitmask_);
  return bitmask_->bytes();
}

template <bool READ>
int32_t Column<READ>::null_count()
{
  if (nullptr == bitmask_) null_count_ = 0;
  if (null_count_ == -1) null_count_ = read_bitmask().count_unset_bits();
  return null_count_;
}

template <bool READ>
detail::Column Column<READ>::view() const
{
  std::vector<detail::Column> children{};
  for (auto child : children_) children.push_back(child.view());

  return detail::Column(code(),
                        column_.is_meta() ? nullptr : raw_column_untyped_read(),
                        num_elements_,
                        nullable() ? raw_bitmask_read() : nullptr,
                        std::move(children));
}

////////////////////////////////////////
// OutputColumn
////////////////////////////////////////

template <typename T>
T *OutputColumn::raw_column() const
{
  return column_.ptr<T>();
}

////////////////////////////////////////
// Deserializers
////////////////////////////////////////

template <bool READ>
void deserialize(Deserializer &ctx, Column<READ> &column)
{
  deserialize(ctx, column.column_);

  bool nullable = false;
  deserialize(ctx, nullable);
  if (nullable) {
    column.bitmask_ = std::make_shared<typename decltype(column.bitmask_)::element_type>();
    deserialize(ctx, *column.bitmask_);
  } else
    column.bitmask_ = nullptr;

  uint32_t num_children = 0;
  deserialize(ctx, num_children);
  column.children_.resize(num_children);

  for (auto &child : column.children_) deserialize(ctx, child);

  if (column.valid()) column.num_elements_ = column.column_.size();
  column.null_count_ = -1;
}

}  // namespace pandas
}  // namespace legate

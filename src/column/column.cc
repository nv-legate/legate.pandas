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

#include "column/column.h"
#include "util/type_dispatch.h"

namespace legate {
namespace pandas {

void deserialize(Deserializer &ctx, OutputColumn &column)
{
  deserialize(ctx, column.column_);

  bool nullable = false;
  deserialize(ctx, nullable);
  if (nullable) {
    column.bitmask_ = std::make_unique<typename decltype(column.bitmask_)::element_type>();
    deserialize(ctx, *column.bitmask_);
  }
  uint32_t num_children;
  deserialize(ctx, num_children);
  column.children_.resize(num_children);
  for (auto &child : column.children_) deserialize(ctx, child);
}

OutputColumn::OutputColumn(OutputColumn &&other) noexcept
  : column_(std::move(other.column_)),
    bitmask_(std::move(other.bitmask_)),
    children_(std::move(other.children_)),
    num_elements_(other.num_elements_)
{
}

OutputColumn &OutputColumn::operator=(OutputColumn &&other) noexcept
{
  column_       = std::move(other.column_);
  bitmask_      = std::move(other.bitmask_);
  children_     = std::move(other.children_);
  num_elements_ = other.num_elements_;
  return *this;
}

void *OutputColumn::raw_column_untyped() const { return column_.untyped_ptr(); }

Bitmask OutputColumn::bitmask() const { return Bitmask(raw_bitmask(), num_elements()); }

std::shared_ptr<Bitmask> OutputColumn::maybe_bitmask() const
{
  return nullptr == bitmask_ ? nullptr : std::make_shared<Bitmask>(raw_bitmask(), num_elements_);
}

Bitmask::AllocType *OutputColumn::raw_bitmask() const
{
#ifdef DEBUG_PANDAS
  assert(nullable());
  assert(valid());
#endif
  return bitmask_->ptr<Bitmask::AllocType>();
}
size_t OutputColumn::elem_size() const { return size_of_type(code()); }

namespace detail {

struct FromScalar {
  template <TypeCode CODE, std::enable_if_t<is_primitive_type<CODE>::value> * = nullptr>
  void operator()(OutputColumn &out, const std::vector<Scalar> &scalars)
  {
    using VAL = pandas_type_of<CODE>;

    out.allocate(scalars.size());

    auto p_out   = out.raw_column<VAL>();
    auto p_out_b = out.raw_bitmask();

    for (auto idx = 0; idx < scalars.size(); ++idx) {
      auto value = scalars[idx].value<VAL>();
      auto valid = scalars[idx].valid();

      p_out[idx]   = value;
      p_out_b[idx] = valid;
    }
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::STRING> * = nullptr>
  void operator()(OutputColumn &out, const std::vector<Scalar> &scalars)
  {
    using VAL = pandas_type_of<CODE>;

    out.allocate(scalars.size());

    size_t num_chars = 0;
    for (auto &scalar : scalars)
      if (scalar.valid()) num_chars += scalar.value<std::string>().size();

    out.child(0).allocate(scalars.size() + 1);
    out.child(1).allocate(num_chars);

    auto p_out_b = out.raw_bitmask();
    auto p_out_o = out.child(0).raw_column<int32_t>();
    auto p_out_c = out.child(1).raw_column<int8_t>();

    int32_t curr_off = 0;
    for (auto idx = 0; idx < scalars.size(); ++idx) {
      auto valid   = scalars[idx].valid();
      p_out_o[idx] = curr_off;
      p_out_b[idx] = valid;

      if (!valid) continue;

      auto &value = scalars[idx].value<std::string>();
      memcpy(&p_out_c[curr_off], value.c_str(), value.size());
      curr_off += value.size();
    }
    p_out_o[scalars.size()] = curr_off;
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::CAT32> * = nullptr>
  void operator()(OutputColumn &out, const std::vector<Scalar> &scalars)
  {
    assert(false);
  }
};

}  // namespace detail

void OutputColumn::return_from_scalars(const std::vector<Scalar> &scalars)
{
  type_dispatch(code(), detail::FromScalar{}, *this, scalars);
}

void OutputColumn::return_from_view(alloc::DeferredBufferAllocator &allocator, detail::Column view)
{
#ifdef DEBUG_PANDAS
  assert(!valid());
#endif
  num_elements_ = view.size();
  auto data     = view.raw_column();
  if (nullptr != data) {
    auto column_buffer = allocator.pop_allocation(data);
    column_.return_from_instance(column_buffer.get_instance(), num_elements_, elem_size());
  } else
    column_.allocate(num_elements_);

  if (nullable()) {
    auto mask = view.raw_bitmask();
    if (mask != nullptr) {
      auto mask_buffer = allocator.pop_allocation(mask);
      bitmask_->return_from_instance(
        mask_buffer.get_instance(), num_elements_, sizeof(Bitmask::AllocType));
    } else {
      bitmask_->allocate(num_elements_);
      if (num_elements_ > 0) bitmask().set_all_valid();
    }
  }

  for (auto idx = 0; idx < num_children() && idx < view.num_children(); ++idx)
    child(idx).return_from_view(allocator, view.child(idx));
}

void OutputColumn::return_column_from_instance(Realm::RegionInstance instance, size_t num_elements)
{
#ifdef DEBUG_PANDAS
  assert(!valid());
#endif
  num_elements_ = num_elements;
  column_.return_from_instance(instance, num_elements, elem_size());
}

void OutputColumn::allocate(size_t num_elements,
                            bool recurse,
                            size_t alignment,
                            size_t bitmask_alignment)
{
  allocate_column(num_elements, alignment);
  allocate_bitmask(num_elements, bitmask_alignment);

  if (recurse) {
    switch (code()) {
      case TypeCode::CAT32: {
        child(0).allocate(num_elements);
        break;
      }
      case TypeCode::STRING: {
        child(0).allocate(num_elements == 0 ? 0 : num_elements + 1);
        break;
      }
      default: {
        break;
      }
    }
  }
}

void OutputColumn::allocate_column(size_t num_elements, size_t alignment)
{
#ifdef DEBUG_PANDAS
  assert(!valid());
#endif
  num_elements_ = num_elements;
  column_.allocate(num_elements, alignment);
}

void OutputColumn::allocate_bitmask(size_t num_elements, size_t alignment)
{
#ifdef DEBUG_PANDAS
  assert(valid() && num_elements_ == num_elements);
#endif
  if (nullable()) bitmask_->allocate(num_elements, alignment);
}

void OutputColumn::make_empty(bool recurse)
{
  allocate(0);
  if (recurse)
    for (auto &child : children_) child.make_empty(recurse);
}

void OutputColumn::copy(const Column<true> &input, bool recurse)
{
  auto size = input.num_elements();
  allocate(size);
  if (size > 0 && !input.is_meta())
    memcpy(raw_column_untyped(), input.raw_column_untyped_read(), input.bytes());
  if (recurse)
    for (auto idx = 0; idx < children_.size(); ++idx)
      children_[idx].copy(input.child(idx), recurse);
}

void OutputColumn::check_all_valid() const
{
  assert(valid());
  for (auto &child : children_) child.check_all_valid();
}

}  // namespace pandas
}  // namespace legate

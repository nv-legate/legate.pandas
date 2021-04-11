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

#include <memory>

#include "pandas.h"
#include "deserializer.h"

#include "bitmask/bitmask.h"
#include "column/region_arg.h"
#include "column/output_region_arg.h"
#include "column/detail/column.h"
#include "scalar/scalar.h"
#include "util/allocator.h"

namespace legate {
namespace pandas {

using PrivilegeMode = Legion::PrivilegeMode;

template <typename FT, PrivilegeMode PRIV>
using Accessor = Legion::
  FieldAccessor<PRIV, FT, 1, Legion::coord_t, Realm::AffineAccessor<FT, 1, Legion::coord_t>>;

template <bool READ>
class Column {
 public:
  template <bool _READ>
  friend void deserialize(Deserializer &ctx, Column<_READ> &column);

 public:
  Column()                          = default;
  Column(const Column<READ> &other) = default;

 public:
  inline auto &child(uint32_t idx) const { return children_[idx]; }
  inline auto num_children() const { return children_.size(); }

 public:
  void set_rect(const Legion::Rect<1> &rect) { column_.set_rect(rect); }

 public:
  inline bool valid() const { return column_.valid(); }
  inline TypeCode code() const { return column_.code; }
  inline bool is_meta() const { return column_.is_meta(); }

 public:
  inline void destroy();

 public:
  template <typename T>
  const AccessorWO<T, 1> &write_accessor() const;
  template <typename T>
  const AccessorRO<T, 1> &read_accessor() const;

 public:
  template <typename T>
  T *raw_column_write() const;
  template <typename T>
  const T *raw_column_read() const;

 public:
  void *raw_column_untyped_write() const;
  const void *raw_column_untyped_read() const;

 public:
  Bitmask::AllocType *raw_bitmask_write() const;
  const Bitmask::AllocType *raw_bitmask_read() const;

 public:
  Bitmask read_bitmask() const;
  Bitmask write_bitmask() const;

 public:
  std::shared_ptr<Bitmask> maybe_read_bitmask() const;
  std::shared_ptr<Bitmask> maybe_write_bitmask() const;

 public:
  inline size_t bytes() const { return column_.bytes(); }
  inline size_t elem_size() const { return column_.elem_size(); }
  inline size_t bitmask_bytes() const;
  inline const Legion::Rect<1> &shape() const { return column_.shape(); }
  inline size_t num_elements() const { return num_elements_; }
  inline bool empty() const { return num_elements_ == 0; }
  int32_t null_count();
  inline bool nullable() const { return nullptr != bitmask_; }

 public:
  detail::Column view() const;

 protected:
  RegionArg<READ> column_{};
  std::shared_ptr<RegionArg<READ>> bitmask_{nullptr};
  std::vector<Column<READ>> children_{};
  size_t num_elements_{0};
  int32_t null_count_{-1};
};

class OutputColumn {
 public:
  friend void deserialize(Deserializer &ctx, OutputColumn &column);

 public:
  OutputColumn()                          = default;
  OutputColumn(const OutputColumn &other) = default;

 public:
  inline auto &child(uint32_t idx) { return children_[idx]; }
  inline auto &child(uint32_t idx) const { return children_[idx]; }
  inline auto num_children() const { return children_.size(); }

 public:
  template <typename T>
  T *raw_column() const;
  void *raw_column_untyped() const;

 public:
  Bitmask bitmask() const;
  std::shared_ptr<Bitmask> maybe_bitmask() const;
  Bitmask::AllocType *raw_bitmask() const;

 public:
  Legion::Rect<1> shape() const { return column_.shape(); }

 public:
  inline size_t num_elements() const { return num_elements_; }
  inline bool empty() const { return num_elements_ == 0; }
  inline bool valid() const { return num_elements_ != -1UL; }
  inline size_t bytes() const { return num_elements_ * elem_size(); }
  size_t elem_size() const;
  inline bool is_meta() const { return column_.is_meta(); }

 public:
  void return_from_scalars(const std::vector<Scalar> &scalars);
  void return_from_view(alloc::DeferredBufferAllocator &allocator, detail::Column view);
  void allocate(size_t num_elements,
                bool recurse             = false,
                size_t alignment         = 16,
                size_t bitmask_alignment = 64);
  void make_empty(bool recurse = true);
  void copy(const Column<true> &input, bool recurse = true);

 public:
  inline TypeCode code() const { return column_.code; }
  inline bool nullable() const { return nullptr != bitmask_; }

 public:
  void destroy();

 protected:
  OutputRegionArg column_{};
  std::shared_ptr<OutputRegionArg> bitmask_{nullptr};
  std::vector<OutputColumn> children_{};
  size_t num_elements_{-1UL};
};

}  // namespace pandas
}  // namespace legate

#include "column/column.inl"

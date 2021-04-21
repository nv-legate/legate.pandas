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
#include "bitmask/bitmask.h"

namespace legate {
namespace pandas {
namespace detail {

class Column {
 public:
  Column(TypeCode code                     = TypeCode::INVALID,
         const void *data                  = nullptr,
         size_t size                       = 0,
         const Bitmask::AllocType *bitmask = nullptr,
         std::vector<Column> &&children    = {})
    : code_(code), data_(data), size_(size), bitmask_(bitmask), children_(children)
  {
  }

 public:
  inline TypeCode code() const { return code_; }
  inline size_t size() const { return size_; }
  inline bool nullable() const { return nullptr != bitmask_; }
  inline size_t num_children() const { return children_.size(); }
  inline size_t bytes() const { return size() * size_of_type(code_); }

 public:
  template <typename T>
  const T *column() const
  {
#ifdef DEBUG_PANDAS
    assert(pandas_type_code_of<T> == to_storage_type_code(code_));
#endif
    return static_cast<const T *>(data_);
  }

  template <typename T>
  T element(const uint32_t idx) const
  {
    return column<T>()[idx];
  }

  const void *raw_column() const { return data_; }

 public:
  inline Bitmask bitmask() const { return Bitmask(bitmask_, size_); }
  inline const Bitmask::AllocType *raw_bitmask() const { return bitmask_; }
  inline std::shared_ptr<Bitmask> maybe_bitmask() const
  {
    return nullptr == bitmask_ ? nullptr : std::make_shared<Bitmask>(bitmask_, size_);
  }

 public:
  inline const Column &child(size_t idx) const { return children_[idx]; }

 protected:
  TypeCode code_{TypeCode::INVALID};
  const void *data_{nullptr};
  size_t size_;
  const Bitmask::AllocType *bitmask_{nullptr};
  std::vector<Column> children_;
};

class Table {
 public:
  Table() = default;
  Table(std::vector<Column> &&columns) : columns_(std::forward<std::vector<Column>>(columns)) {}

 public:
  Table(Table &&other)      = default;
  Table(const Table &table) = default;

  Table &operator=(Table &&other) = default;
  Table &operator=(const Table &table) = default;

 public:
  inline size_t size() const { return columns_.front().size(); }
  inline size_t num_columns() const { return columns_.size(); }

 public:
  inline const std::vector<Column> &columns() const { return columns_; }
  inline decltype(auto) column(size_t idx) const { return columns_[idx]; }

 public:
  std::vector<Column> select(std::vector<int32_t> indices) const
  {
    std::vector<Column> selected;
    for (auto &idx : indices) selected.push_back(columns_[idx]);
    return std::move(selected);
  }

 public:
  std::vector<Column> release() { return std::move(columns_); }

 protected:
  std::vector<Column> columns_;
};

template <>
inline std::string Column::element(const uint32_t idx) const
{
#ifdef DEBUG_PANDAS
  assert(TypeCode::STRING == code_);
#endif
  auto p_offsets = child(0).column<int32_t>();
  auto p_chars   = child(1).column<int8_t>();
  return std::string(&p_chars[p_offsets[idx]], &p_chars[p_offsets[idx + 1]]);
}

}  // namespace detail
}  // namespace pandas
}  // namespace legate

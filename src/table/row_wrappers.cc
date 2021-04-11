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

#include "table/row_wrappers.h"
#include "column/detail/column.h"
#include "util/zip_for_each.h"

#include <limits>

namespace legate {
namespace pandas {
namespace table {

bool Row::all_valid() const
{
  for (auto col_idx = 0; col_idx < columns_.size(); ++col_idx) {
    const auto &column = columns_[col_idx];
    if (column.nullable() && !column.bitmask().get(idx_)) return false;
  }
  return true;
}

using ColumnView = legate::pandas::detail::Column;

namespace detail {

struct Hasher {
  template <
    TypeCode CODE,
    std::enable_if_t<is_primitive_type<CODE>::value || CODE == TypeCode::STRING> * = nullptr>
  size_t operator()(const ColumnView &column, size_t idx, size_t hash)
  {
    using T = pandas_type_of<CODE>;
    if (column.nullable() && !column.bitmask().get(idx))
      return std::hash<int32_t>{}(std::numeric_limits<int32_t>::max()) ^ (hash << 1);
    else
      return std::hash<T>{}(column.element<T>(idx)) ^ (hash << 1);
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::CAT32> * = nullptr>
  size_t operator()(const ColumnView &column, size_t idx, size_t hash)
  {
    using T = pandas_type_of<CODE>;
    if (column.nullable() && !column.bitmask().get(idx))
      return std::hash<int32_t>{}(std::numeric_limits<int32_t>::max()) ^ (hash << 1);
    else {
      const auto code = column.child(0).element<uint32_t>(idx);
      return std::hash<std::string>{}(column.child(1).element<std::string>(code)) ^ (hash << 1);
    }
  }

  template <TypeCode CODE,
            std::enable_if_t<!is_primitive_type<CODE>::value && CODE != TypeCode::STRING &&
                             CODE != TypeCode::CAT32> * = nullptr>
  size_t operator()(const ColumnView &column, size_t idx, size_t hash)
  {
    assert(false);
    return hash;
  }
};

struct Equal {
  template <
    TypeCode CODE,
    std::enable_if_t<is_primitive_type<CODE>::value || CODE == TypeCode::STRING> * = nullptr>
  bool operator()(const ColumnView &c1, const ColumnView &c2, size_t idx1, size_t idx2)
  {
    using T = pandas_type_of<CODE>;
    return c1.element<T>(idx1) == c2.element<T>(idx2);
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::CAT32> * = nullptr>
  bool operator()(const ColumnView &c1, const ColumnView &c2, size_t idx1, size_t idx2)
  {
    using T          = pandas_type_of<CODE>;
    const auto code1 = c1.child(0).element<uint32_t>(idx1);
    const auto code2 = c2.child(0).element<uint32_t>(idx2);
    return c1.child(1).element<std::string>(code1) == c2.child(1).element<std::string>(code2);
  }

  template <TypeCode CODE,
            std::enable_if_t<!is_primitive_type<CODE>::value && CODE != TypeCode::STRING &&
                             CODE != TypeCode::CAT32> * = nullptr>
  bool operator()(const ColumnView &c1, const ColumnView &c2, size_t idx1, size_t idx2)
  {
    assert(false);
    return false;
  }
};

enum class CmpResult : int32_t {
  YES   = 0,
  NO    = 1,
  EQUAL = 2,
};

template <typename T>
CmpResult compare_values(const T &v1, const T &v2, bool asc)
{
  if (asc) {
    std::less<T> op{};
    if (op(v1, v2))
      return CmpResult::YES;
    else if (op(v2, v1))
      return CmpResult::NO;
  } else {
    std::greater<T> op{};
    if (op(v1, v2))
      return CmpResult::YES;
    else if (op(v2, v1))
      return CmpResult::NO;
  }
  return CmpResult::EQUAL;
}

struct Compare {
  template <
    TypeCode CODE,
    std::enable_if_t<is_primitive_type<CODE>::value || CODE == TypeCode::STRING> * = nullptr>
  CmpResult operator()(
    const ColumnView &c1, const ColumnView &c2, size_t idx1, size_t idx2, bool asc)
  {
    using T       = pandas_type_of<CODE>;
    const auto v1 = c1.element<T>(idx1);
    const auto v2 = c2.element<T>(idx2);
    return compare_values(v1, v2, asc);
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::CAT32> * = nullptr>
  CmpResult operator()(
    const ColumnView &c1, const ColumnView &c2, size_t idx1, size_t idx2, bool asc)
  {
    using T          = pandas_type_of<CODE>;
    const auto code1 = c1.child(0).element<uint32_t>(idx1);
    const auto code2 = c2.child(0).element<uint32_t>(idx2);
    const auto v1    = c1.child(1).element<std::string>(code1);
    const auto v2    = c2.child(1).element<std::string>(code2);
    return compare_values(v1, v2, asc);
  }

  template <TypeCode CODE,
            std::enable_if_t<!is_primitive_type<CODE>::value && CODE != TypeCode::STRING &&
                             CODE != TypeCode::CAT32> * = nullptr>
  CmpResult operator()(
    const ColumnView &c1, const ColumnView &c2, size_t idx1, size_t idx2, bool asc)
  {
    assert(false);
    return CmpResult::EQUAL;
  }
};

bool compare_rows(const Row &r1,
                  const Row &r2,
                  const std::vector<bool> &ascending,
                  bool put_null_first)
{
#ifdef DEBUG_PANDAS
  assert(r1.columns_.size() == r2.columns_.size());
#endif

  auto l = r1.idx_;
  auto r = r2.idx_;

  for (auto col_idx = 0; col_idx < r1.columns_.size(); ++col_idx) {
    auto &c1 = r1.columns_[col_idx];
    auto &c2 = r2.columns_[col_idx];
    auto asc = ascending[col_idx];

    bool valid1 = true;
    bool valid2 = true;

    if (c1.nullable()) valid1 = c1.bitmask().get(l);
    if (c2.nullable()) valid2 = c2.bitmask().get(r);

    switch (compare_values(valid1, valid2, asc)) {
      case CmpResult::YES: return put_null_first == asc;
      case CmpResult::NO: return put_null_first != asc;
      case CmpResult::EQUAL: break;
    }

    if (!valid1) continue;

#ifdef DEBUG_PANDAS
    assert(c1.code() == c2.code());
#endif

    switch (type_dispatch(c1.code(), Compare{}, c1, c2, l, r, asc)) {
      case CmpResult::YES: return true;
      case CmpResult::NO: return false;
      case CmpResult::EQUAL: break;
    }
  }
  return false;
}

}  // namespace detail

size_t RowHasher::operator()(const Row &r) const noexcept
{
  size_t hash = 0xFFFFFFFFFFFFFFFFUL;
  for (const auto &column : r.columns_)
    hash = type_dispatch(column.code(), detail::Hasher{}, column, r.idx_, hash);
  return hash;
}

bool RowEqual::operator()(const Row &r1, const Row &r2) const noexcept
{
#ifdef DEBUG_PANDAS
  assert(r1.size() == r2.size());
#endif
  for (auto col_idx = 0; col_idx < r1.columns_.size(); ++col_idx) {
    const auto &c1 = r1.columns_[col_idx];
    const auto &c2 = r2.columns_[col_idx];

#ifdef DEBUG_PANDAS
    assert(c1.code() == c2.code());
#endif

    bool c1_valid = true;
    bool c2_valid = true;

    if (c1.nullable()) c1_valid = c1.bitmask().get(r1.idx_);
    if (c2.nullable()) c2_valid = c2.bitmask().get(r2.idx_);

    if (c1_valid != c2_valid) return false;
    if (!c1_valid) continue;

    auto equal = type_dispatch(c1.code(), detail::Equal{}, c1, c2, r1.idx_, r2.idx_);
    if (!equal) return false;
  }
  return true;
};

bool RowCompare::operator()(const int64_t &l, const int64_t &r) const noexcept
{
  return detail::compare_rows(Row{columns_, static_cast<size_t>(l)},
                              Row{columns_, static_cast<size_t>(r)},
                              ascending_,
                              put_null_first_);
}

}  // namespace table
}  // namespace pandas
}  // namespace legate

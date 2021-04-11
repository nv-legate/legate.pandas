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
#include "column/detail/column.h"

namespace legate {
namespace pandas {
namespace table {

struct Row {
  inline size_t size() const noexcept { return columns_.size(); }
  bool all_valid() const;

  const std::vector<detail::Column>& columns_;
  size_t idx_;
};

struct RowHasher {
  size_t operator()(const Row& r) const noexcept;
};

struct RowEqual {
  bool operator()(const Row& r1, const Row& r2) const noexcept;
};

struct RowLess {
  RowLess(const std::vector<detail::Column>& columns, bool put_null_first)
    : columns_(columns), put_null_first_(put_null_first)
  {
  }
  bool operator()(const int64_t& l, const int64_t& r) const noexcept;

  const std::vector<detail::Column>& columns_;
  bool put_null_first_;
};

struct RowGreater {
  RowGreater(const std::vector<detail::Column>& columns, bool put_null_first)
    : columns_(columns), put_null_first_(put_null_first)
  {
  }
  bool operator()(const int64_t& l, const int64_t& r) const noexcept;

  const std::vector<detail::Column>& columns_;
  bool put_null_first_;
};

struct RowCompare {
  RowCompare(const std::vector<detail::Column>& columns,
             const std::vector<bool>& ascending,
             bool put_null_first)
    : columns_(columns), ascending_(ascending), put_null_first_(put_null_first)
  {
  }
  bool operator()(const int64_t& l, const int64_t& r) const noexcept;

  const std::vector<detail::Column>& columns_;
  const std::vector<bool>& ascending_;
  bool put_null_first_;
};

namespace detail {

bool compare_rows(const Row& r1,
                  const Row& r2,
                  const std::vector<bool>& ascending,
                  bool put_null_first);

}  // namespace detail

}  // namespace table
}  // namespace pandas
}  // namespace legate

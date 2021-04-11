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

#include "index/search.h"
#include "column/column.h"
#include "util/type_dispatch.h"

namespace legate {
namespace pandas {
namespace index {

using namespace Legion;
using ColumnView = detail::Column;

namespace detail {

struct Search {
  template <TypeCode CODE, std::enable_if_t<is_primitive_type<CODE>::value> * = nullptr>
  Scalar operator()(const ColumnView &in, const Scalar &to_find, bool forward)
  {
    using VAL = pandas_type_of<CODE>;

    int64_t size = static_cast<int64_t>(in.size());
    auto p_in    = in.column<VAL>();
    auto value   = to_find.value<VAL>();

    if (in.nullable()) {
      auto in_b = in.bitmask();

      if (forward) {
        for (int64_t idx = 0; idx < size; ++idx) {
          if (!in_b.get(idx)) continue;
          if (p_in[idx] == value) return Scalar(true, idx);
        }
      } else {
        for (int64_t idx = size - 1; idx >= 0; --idx) {
          if (!in_b.get(idx)) continue;
          if (p_in[idx] == value) return Scalar(true, idx);
        }
      }
    } else {
      if (forward) {
        for (int64_t idx = 0; idx < size; ++idx)
          if (p_in[idx] == value) return Scalar(true, idx);
      } else {
        for (int64_t idx = size - 1; idx >= 0; --idx)
          if (p_in[idx] == value) return Scalar(true, idx);
      }
    }
    return Scalar(TypeCode::INT64);
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::STRING> * = nullptr>
  Scalar operator()(const ColumnView &in, const Scalar &to_find, bool forward)
  {
    int64_t size      = static_cast<int64_t>(in.size());
    std::string value = to_find.value<std::string>();

    if (in.nullable()) {
      auto in_b = in.bitmask();

      if (forward) {
        for (int64_t idx = 0; idx < size; ++idx) {
          if (!in_b.get(idx)) continue;
          if (in.element<std::string>(idx) == value) return Scalar(true, idx);
        }
      } else {
        for (int64_t idx = size - 1; idx >= 0; --idx) {
          if (!in_b.get(idx)) continue;
          if (in.element<std::string>(idx) == value) return Scalar(true, idx);
        }
      }
    } else {
      if (forward) {
        for (int64_t idx = 0; idx < size; ++idx)
          if (in.element<std::string>(idx) == value) return Scalar(true, idx);
      } else {
        for (int64_t idx = size - 1; idx >= 0; --idx)
          if (in.element<std::string>(idx) == value) return Scalar(true, idx);
      }
    }

    return Scalar(TypeCode::INT64);
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::CAT32> * = nullptr>
  Scalar operator()(const ColumnView &in, const Scalar &to_find, bool forward)
  {
    assert(false);
    return Scalar(TypeCode::INT64);
  }
};

}  // namespace detail

Scalar search(const ColumnView &in, const Scalar &to_find, bool forward)
{
  if (!to_find.valid()) return Scalar(TypeCode::INT64);
  return type_dispatch(in.code(), detail::Search{}, in, to_find, forward);
}

}  // namespace index
}  // namespace pandas
}  // namespace legate

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

#include "pandas.h"
#include "bitmask/bitmask.h"
#include "column/detail/column.h"

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace legate {
namespace pandas {
namespace copy {
namespace detail {

template <typename VAL>
struct broadcast_fn : public thrust::unary_function<coord_t, VAL> {
  using value_type = VAL;

  broadcast_fn(const VAL &value) : value_(value) {}

  VAL operator()(coord_t idx) { return value_; }

  VAL value_{};
};

template <typename VAL>
struct accessor_fn : public thrust::unary_function<coord_t, VAL> {
  using value_type = VAL;

  accessor_fn(const pandas::detail::Column &column) : column_(column) {}

  VAL operator()(coord_t idx) { return column_.element<VAL>(idx); }

  const pandas::detail::Column &column_;
};

struct bitmask_accessor_fn : public thrust::unary_function<coord_t, Bitmask::AllocType> {
  using value_type = Bitmask::AllocType;

  bitmask_accessor_fn(const Bitmask &bitmask) : bitmask_(bitmask) {}

  Bitmask::AllocType operator()(coord_t idx) { return bitmask_.get(idx); }

  const Bitmask &bitmask_;
};

struct size_accessor_fn : public thrust::unary_function<coord_t, int32_t> {
  using value_type = int32_t;

  size_accessor_fn(const pandas::detail::Column &column) : column_(column) {}

  int32_t operator()(coord_t idx)
  {
    int32_t size = column_.element<int32_t>(idx + 1) - column_.element<int32_t>(idx);
#ifdef DEBUG_PANDAS
    assert(size >= 0);
#endif
    return size;
  }

  const pandas::detail::Column &column_;
};

}  // namespace detail
}  // namespace copy
}  // namespace pandas
}  // namespace legate

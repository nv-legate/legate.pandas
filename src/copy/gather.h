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
#include "column/detail/column.h"
#include "util/allocator.h"

namespace legate {
namespace pandas {
namespace copy {

enum class OutOfRangePolicy : int {
  NULLIFY = 0,
  IGNORE  = 1,
};

detail::Column gather(const detail::Column &input,
                      const std::vector<int64_t> &mapping,
                      bool has_out_of_range,
                      OutOfRangePolicy policy,
                      alloc::Allocator &allocator);

detail::Column gather(const Legion::Rect<1> &rect,
                      const std::vector<int64_t> &mapping,
                      bool has_out_of_range,
                      OutOfRangePolicy policy,
                      alloc::Allocator &allocator);

}  // namespace copy
}  // namespace pandas
}  // namespace legate

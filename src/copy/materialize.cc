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

#include "copy/materialize.h"

namespace legate {
namespace pandas {
namespace copy {

detail::Column materialize(const Legion::Rect<1> &rect,
                           int64_t start,
                           int64_t step,
                           alloc::Allocator &allocator)
{
  auto size  = rect.volume();
  auto p_out = allocator.allocate_elements<int64_t>(size);
  for (int64_t idx = 0; idx < size; ++idx) p_out[idx] = start + (rect.lo[0] + idx) * step;
  return detail::Column(TypeCode::INT64, p_out, size);
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

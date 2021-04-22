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

#include <assert.h>
#include <string.h>

#include "bitmask/bitmask.h"

namespace legate {
namespace pandas {

Bitmask::Bitmask(AllocType *bitmask, size_t num_elements)
  : bitmask(bitmask), num_elements(num_elements)
{
}

Bitmask::Bitmask(const AllocType *bitmask, size_t num_elements)
  : bitmask(const_cast<AllocType *>(bitmask)), num_elements(num_elements)
{
}

Bitmask::Bitmask(size_t num_elements, alloc::Allocator &allocator, bool init, bool init_value)
  : bitmask(allocator.allocate_elements<AllocType>(num_elements)), num_elements(num_elements)
{
  if (init) {
    if (init_value)
      set_all_valid();
    else
      clear();
  }
}

void Bitmask::set_all_valid(void) { memset(bitmask, 0x01, num_elements); }

void Bitmask::clear(void) { memset(bitmask, 0, num_elements); }

size_t Bitmask::count_set_bits(void) const
{
  size_t result = 0;
  for (size_t idx = 0; idx < num_elements; ++idx) result += get(idx);
  return result;
}

void Bitmask::copy(const Bitmask &target) const
{
  assert(num_elements == target.num_elements);
  memcpy(target.bitmask, bitmask, num_elements);
}

void intersect_bitmasks(Bitmask &out, const Bitmask &in1, const Bitmask &in2)
{
  assert(out.num_elements == in1.num_elements && out.num_elements == in2.num_elements);
  for (size_t idx = 0; idx < out.num_elements; ++idx) out.set(idx, in1.get(idx) && in2.get(idx));
}

}  // namespace pandas
}  // namespace legate

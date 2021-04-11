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

#include "bitmask/compact_bitmask.h"

namespace legate {
namespace pandas {

CompactBitmask::CompactBitmask(AllocType *bitmask, size_t num_elements)
  : bitmask(bitmask), num_elements(num_elements), size((num_elements + NUM_BITS - 1) / NUM_BITS)
{
}

CompactBitmask::CompactBitmask(const AllocType *bitmask, size_t num_elements)
  : bitmask(const_cast<AllocType *>(bitmask)),
    num_elements(num_elements),
    size((num_elements + NUM_BITS - 1) / NUM_BITS)
{
}

CompactBitmask::CompactBitmask(size_t num_elements, alloc::Allocator &allocator)
  : num_elements(num_elements), size(((num_elements + NUM_BITS - 1) / NUM_BITS + 63) / 64 * 64)
{
  bitmask = allocator.allocate_elements<AllocType>(size);
}

}  // namespace pandas
}  // namespace legate

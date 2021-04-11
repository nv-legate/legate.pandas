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

#include "bitmask/compact_bitmask.h"
#include "cudf_util/bitmask.h"

namespace legate {
namespace pandas {

CompactBitmask::CompactBitmask(cudf::bitmask_type *bitmask, size_t num_elements)
  : bitmask(reinterpret_cast<AllocType *>(bitmask)),
    num_elements(num_elements),
    size((num_elements + NUM_BITS - 1) / NUM_BITS)
{
}

CompactBitmask::CompactBitmask(const cudf::bitmask_type *bitmask, size_t num_elements)
  : bitmask(reinterpret_cast<AllocType *>(const_cast<cudf::bitmask_type *>(bitmask))),
    num_elements(num_elements),
    size((num_elements + NUM_BITS - 1) / NUM_BITS)
{
}

void CompactBitmask::to_boolmask(const Bitmask &target, cudaStream_t stream) const
{
  util::to_boolmask(target.raw_ptr(), bitmask, target.num_elements, stream);
}

}  // namespace pandas
}  // namespace legate

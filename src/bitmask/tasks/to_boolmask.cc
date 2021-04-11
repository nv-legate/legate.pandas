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

#include "bitmask/tasks/to_boolmask.h"
#include "bitmask/bitmask.h"

namespace legate {
namespace pandas {
namespace bitmask {

using namespace Legion;

/*static*/ void ToBoolmaskTask::cpu_variant(const Task *task,
                                            const std::vector<PhysicalRegion> &regions,
                                            Context context,
                                            Runtime *runtime)
{
  LegateDeserializer derez(task->args, task->arglen);

  const uint32_t bit_idx = derez.unpack_32bit_uint();
  auto bitmask_acc       = derez.unpack_accessor_RO<Bitmask::AllocType, 1>(regions[bit_idx]);
  const Rect<1> bitmask_shape(regions[0]);

  const uint32_t bool_idx = derez.unpack_32bit_uint();
  auto boolmask_acc       = derez.unpack_accessor_WO<Bitmask::AllocType, 1>(regions[bool_idx]);
  const Rect<1> boolmask_shape(regions[1]);
  size_t size = boolmask_shape.volume();

  Bitmask bitmask(bitmask_acc.ptr(bitmask_shape.lo), size);
  Bitmask::AllocType *boolmask = boolmask_acc.ptr(boolmask_shape.lo);

  for (unsigned idx = 0; idx < size; ++idx)
    boolmask[idx] = static_cast<Bitmask::AllocType>(bitmask.get(idx));
}

static void __attribute__((constructor)) register_tasks(void)
{
  ToBoolmaskTask::register_variants();
}

}  // namespace bitmask
}  // namespace pandas
}  // namespace legate

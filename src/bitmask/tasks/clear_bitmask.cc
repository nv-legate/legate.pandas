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

#include "bitmask/tasks/clear_bitmask.h"
#include "bitmask/bitmask.h"

namespace legate {
namespace pandas {
namespace bitmask {

using namespace Legion;

/*static*/ void ClearBitmaskTask::cpu_variant(const Task *task,
                                              const std::vector<PhysicalRegion> &regions,
                                              Context context,
                                              Runtime *runtime)
{
  LegateDeserializer derez(task->args, task->arglen);

  const Rect<1> rect(regions[0]);

  const unsigned idx = derez.unpack_32bit_uint();
  auto bitmask_acc   = derez.unpack_accessor_RO<Bitmask::AllocType, 1>(regions[idx]);
  const Rect<1> bitmask_rect(regions[idx]);

  if (rect.empty()) return;

  Bitmask bitmask(bitmask_acc.ptr(bitmask_rect.lo), rect.volume());
  bitmask.set_all_valid();
}

static void __attribute__((constructor)) register_tasks(void)
{
  ClearBitmaskTask::register_variants();
}

}  // namespace bitmask
}  // namespace pandas
}  // namespace legate

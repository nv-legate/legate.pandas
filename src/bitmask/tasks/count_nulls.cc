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

#include "bitmask/tasks/count_nulls.h"
#include "bitmask/bitmask.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace bitmask {

using namespace Legion;

/*static*/ uint64_t CountNullsTask::cpu_variant(const Task *task,
                                                const std::vector<PhysicalRegion> &regions,
                                                Context context,
                                                Runtime *runtime)
{
  Deserializer derez{task, regions};

  AccessorRO<Bitmask::AllocType, 1> bitmask_acc;
  const auto bitmask_rect = deserialize(derez, bitmask_acc);
  const Bitmask bitmask(bitmask_acc.ptr(bitmask_rect.lo), bitmask_rect.volume());
  return bitmask.count_unset_bits();
}

static void __attribute__((constructor)) register_tasks(void)
{
  CountNullsTask::register_variants_with_return<uint64_t, uint64_t>();
}

}  // namespace bitmask
}  // namespace pandas
}  // namespace legate

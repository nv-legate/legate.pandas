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

#include <math.h>

#include "bitmask/tasks/to_boolmask.h"
#include "bitmask/bitmask.h"
#include "cudf_util/bitmask.h"
#include "util/gpu_task_context.h"

namespace legate {
namespace pandas {
namespace bitmask {

using namespace Legion;

/*static*/ void ToBoolmaskTask::gpu_variant(const Task *task,
                                            const std::vector<PhysicalRegion> &regions,
                                            Context context,
                                            Runtime *runtime)
{
  LegateDeserializer derez(task->args, task->arglen);

  const uint32_t bit_idx = derez.unpack_32bit_uint();
  auto bitmask_acc       = derez.unpack_accessor_RO<Bitmask::AllocType, 1>(regions[bit_idx]);
  const Rect<1> bitmask_shape(regions[bit_idx]);

  const uint32_t bool_idx = derez.unpack_32bit_uint();
  auto boolmask_acc       = derez.unpack_accessor_WO<Bitmask::AllocType, 1>(regions[bool_idx]);
  const Rect<1> boolmask_shape(regions[bool_idx]);
  size_t num_bits = boolmask_shape.volume();

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  util::to_boolmask(
    boolmask_acc.ptr(boolmask_shape.lo), bitmask_acc.ptr(bitmask_shape.lo), num_bits, stream);
}

}  // namespace bitmask
}  // namespace pandas
}  // namespace legate

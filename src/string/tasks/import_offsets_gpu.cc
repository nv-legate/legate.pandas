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

#include "string/tasks/import_offsets.h"
#include "bitmask/bitmask.h"
#include "bitmask/compact_bitmask.h"
#include "column/column.h"
#include "util/gpu_task_context.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace string {

using namespace Legion;

/*static*/ void ImportOffsetsTask::gpu_variant(const Task *task,
                                               const std::vector<PhysicalRegion> &regions,
                                               Context context,
                                               Runtime *runtime)
{
  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();
  Deserializer ctx{task, regions};

  Column<false> out;
  RegionArg<true> in_offsets_arg;

  deserialize(ctx, out);
  deserialize(ctx, in_offsets_arg);

  auto *out_offsets = out.child(0).raw_column_write<int32_t>();
  auto *in_offsets  = in_offsets_arg.raw_read<int32_t>();

  const auto num_offsets = out.child(0).num_elements();
  assert(in_offsets_arg.size() == num_offsets);

  cudaMemcpyAsync(
    out_offsets, in_offsets, sizeof(int32_t) * num_offsets, cudaMemcpyDeviceToDevice, stream);

  if (out.nullable()) {
    const auto size = num_offsets - 1;
    RegionArg<true> in_bitmask_arg;
    deserialize(ctx, in_bitmask_arg);

    assert(in_bitmask_arg.size() == (size + 7) / 8);

    CompactBitmask in_b(in_bitmask_arg.raw_read<CompactBitmask::AllocType>(), size);
    Bitmask out_b = out.write_bitmask();
    in_b.to_boolmask(out_b, stream);
  }
}

}  // namespace string
}  // namespace pandas
}  // namespace legate

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

#include "range/offsets_to_ranges.h"
#include "column/column.h"
#include "util/cuda_helper.h"
#include "util/gpu_task_context.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace range {

using namespace Legion;

namespace detail {

__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  to_ranges(const size_t max, Rect<1> *out, const int32_t *in, coord_t offset)
{
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= max) return;
  auto lo = static_cast<coord_t>(in[i]) + offset;
  auto hi = static_cast<coord_t>(in[i + 1]) - 1 + offset;
  out[i]  = Rect<1>{lo, hi};
}

}  // namespace detail

/*static*/ void OffsetsToRangesTask::gpu_variant(const Task *task,
                                                 const std::vector<PhysicalRegion> &regions,
                                                 Context context,
                                                 Runtime *runtime)
{
  Deserializer ctx{task, regions};

  Column<false> out;
  Column<true> in;
  Column<true> chars;

  deserialize(ctx, out);
  deserialize(ctx, in);
  deserialize(ctx, chars);

  const auto num_offsets = in.num_elements();

  if (num_offsets == 0) return;

#ifdef DEBUG_PANDAS
  assert(num_offsets > 1);
#endif

  GPUTaskContext gpu_ctx{};

  const auto num_ranges = num_offsets - 1;

  auto *out_ranges = out.raw_column_write<Rect<1>>();
  auto *in_offsets = in.raw_column_read<int32_t>();
  auto offset      = chars.shape().lo[0];

  const auto blocks = (num_ranges + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  detail::to_ranges<<<blocks, THREADS_PER_BLOCK, 0, gpu_ctx.stream()>>>(
    num_ranges, out_ranges, in_offsets, offset);
}

}  // namespace range
}  // namespace pandas
}  // namespace legate

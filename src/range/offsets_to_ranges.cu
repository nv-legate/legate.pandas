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

#include <thrust/transform.h>

#include <rmm/exec_policy.hpp>

namespace legate {
namespace pandas {
namespace range {

using namespace Legion;

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

  const int64_t num_offsets = static_cast<int64_t>(in.num_elements());

  if (num_offsets == 0) return;

#ifdef DEBUG_PANDAS
  assert(num_offsets > 1);
#endif

  GPUTaskContext gpu_ctx{};

  const auto num_ranges = num_offsets - 1;

  auto *out_ranges = out.raw_column_write<Rect<1>>();
  auto *in_offsets = in.raw_column_read<int32_t>();
  auto offset      = chars.shape().lo[0];

  auto start = thrust::make_counting_iterator<int64_t>(0);
  auto stop  = thrust::make_counting_iterator<int64_t>(num_ranges);

  thrust::transform(rmm::exec_policy(gpu_ctx.stream()),
                    start,
                    stop,
                    out_ranges,
                    [in_offsets, offset] __device__(auto idx) {
                      return Rect<1>(in_offsets[idx] + offset, in_offsets[idx + 1] + offset - 1);
                    });
}

}  // namespace range
}  // namespace pandas
}  // namespace legate

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

#include "range/ranges_to_offsets.h"
#include "column/column.h"
#include "util/cuda_helper.h"
#include "util/gpu_task_context.h"
#include "deserializer.h"

#include <thrust/scan.h>
#include <thrust/transform.h>

#include <rmm/exec_policy.hpp>

namespace legate {
namespace pandas {
namespace range {

using namespace Legion;

/*static*/ int64_t RangesToOffsetsTask::gpu_variant(const Task *task,
                                                    const std::vector<PhysicalRegion> &regions,
                                                    Context context,
                                                    Runtime *runtime)
{
  Deserializer ctx{task, regions};

  OutputColumn out;
  Column<true> in;

  deserialize(ctx, out);
  deserialize(ctx, in);

  int64_t size = static_cast<int64_t>(in.num_elements());
  if (size == 0) {
    out.allocate(0);
    return 0;
  }

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  out.allocate(size + 1);

  auto *out_offsets = out.raw_column<int32_t>();
  auto *in_ranges   = in.raw_column_read<Rect<1>>();

  auto start = thrust::make_counting_iterator<int64_t>(0);
  auto stop  = thrust::make_counting_iterator<int64_t>(size + 1);

  thrust::transform(
    rmm::exec_policy(stream), start, stop, out_offsets, [in_ranges, size] __device__(auto idx) {
      return idx == size ? 0 : in_ranges[idx].volume();
    });

  thrust::exclusive_scan(
    rmm::exec_policy(stream), out_offsets, out_offsets + size + 1, out_offsets);

  return size + 1;
}

}  // namespace range
}  // namespace pandas
}  // namespace legate

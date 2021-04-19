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

#include "copy/tasks/slice_by_range.h"
#include "cudf_util/allocators.h"
#include "cudf_util/column.h"
#include "util/gpu_task_context.h"

#include <cudf/copying.hpp>

namespace legate {
namespace pandas {
namespace copy {

using namespace Legion;

using SliceByRangeArg = SliceByRangeTask::SliceByRangeTaskArgs::SliceByRangeArg;

static inline OutputColumn &output(SliceByRangeArg &arg) { return arg.first; };

static inline Column<true> &input(SliceByRangeArg &arg) { return arg.second; };

/*static*/ int64_t SliceByRangeTask::gpu_variant(const Task *task,
                                                 const std::vector<PhysicalRegion> &regions,
                                                 Context context,
                                                 Runtime *runtime)
{
  Deserializer ctx{task, regions};

  SliceByRangeTaskArgs args;
  deserialize(ctx, args);

  auto bounds = input(args.pairs.front()).shape();
  auto range  = bounds.intersection(args.range);

  int64_t out_size = static_cast<int64_t>(range.volume());
  if (out_size == 0) {
    for (auto &pair : args.pairs) output(pair).make_empty(true);
    return 0;
  }

  std::vector<cudf::size_type> indices{
    static_cast<cudf::size_type>(range.lo[0] - bounds.lo[0]),
    static_cast<cudf::size_type>(range.hi[0] - bounds.lo[0] + 1)};

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  DeferredBufferAllocator mr;

  for (auto &pair : args.pairs) {
    auto &in    = input(pair);
    auto &out   = output(pair);
    auto slices = cudf::slice(to_cudf_column(in, stream), indices);
    auto slice  = std::make_unique<cudf::column>(slices[0], stream, &mr);
    from_cudf_column(out, std::move(slice), stream, mr);
  }

  return out_size;
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

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

#include "sorting/tasks/sample_keys.h"
#include "sorting/utilities.h"
#include "cudf_util/column.h"
#include "util/gpu_task_context.h"

#include <cudf/detail/copy.hpp>

namespace legate {
namespace pandas {
namespace sorting {

using namespace Legion;

/*static*/ void SampleKeysTask::gpu_variant(const Task *task,
                                            const std::vector<PhysicalRegion> &regions,
                                            Context context,
                                            Runtime *runtime)
{
  Deserializer ctx{task, regions};

  SampleKeysArgs args;
  deserialize(ctx, args);

  auto size = args.input[0].num_elements();

  if (0 == size) {
    for (auto &column : args.output) column.make_empty(true);
    return;
  }

  auto num_samples = std::min<size_t>(32, std::max<size_t>(size / 4, 1));

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  std::vector<cudf::column_view> keys;
  for (auto &column : args.input) keys.push_back(to_cudf_column(column, stream));

  DeferredBufferAllocator mr;
  auto samples = cudf::detail::sample(cudf::table_view(std::move(keys)),
                                      num_samples,
                                      cudf::sample_with_replacement::FALSE,
                                      Realm::Clock::current_time_in_nanoseconds(),
                                      stream,
                                      &mr);

  from_cudf_table(args.output, std::move(samples), stream, mr);
}

}  // namespace sorting
}  // namespace pandas
}  // namespace legate

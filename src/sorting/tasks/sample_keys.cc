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
#include "column/detail/column.h"
#include "copy/gather.h"
#include "util/allocator.h"
#include "util/zip_for_each.h"

namespace legate {
namespace pandas {
namespace sorting {

using namespace Legion;

using SampleKeysArgs = SampleKeysTask::SampleKeysArgs;

void SampleKeysTask::SampleKeysArgs::sanity_check(void)
{
  for (auto &column : input) assert(input[0].shape() == column.shape());
  util::for_each(input, output, [&](auto &in, auto &out) { assert(in.code() == out.code()); });
}

/*static*/ void SampleKeysTask::cpu_variant(const Task *task,
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

  std::vector<int64_t> mapping;
  sample(size, mapping);
  alloc::DeferredBufferAllocator allocator;
  util::for_each(args.output, args.input, [&](auto &output, auto &input) {
    auto &&gathered =
      copy::gather(input.view(), mapping, false, copy::OutOfRangePolicy::IGNORE, allocator);
    output.return_from_view(allocator, gathered);
  });
}

void deserialize(Deserializer &ctx, SampleKeysTask::SampleKeysArgs &args)
{
  uint32_t num_columns = 0;
  deserialize(ctx, num_columns);
  args.input.resize(num_columns);
  args.output.resize(num_columns);
  deserialize(ctx, args.input, false);
  deserialize(ctx, args.output, false);
}

static void __attribute__((constructor)) register_tasks(void)
{
  SampleKeysTask::register_variants();
}

}  // namespace sorting
}  // namespace pandas
}  // namespace legate

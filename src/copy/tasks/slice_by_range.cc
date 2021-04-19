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
#include "copy/gather.h"
#include "util/allocator.h"

namespace legate {
namespace pandas {
namespace copy {

using namespace Legion;

using SliceByRangeArg = SliceByRangeTask::SliceByRangeTaskArgs::SliceByRangeArg;

static inline OutputColumn &output(SliceByRangeArg &arg) { return arg.first; };

static inline Column<true> &input(SliceByRangeArg &arg) { return arg.second; };

void SliceByRangeTask::SliceByRangeTaskArgs::sanity_check(void)
{
  for (auto &pair : pairs)
    assert(!input(pair).valid() || output(pair).code() == input(pair).code());
}

/*static*/ int64_t SliceByRangeTask::cpu_variant(const Task *task,
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

  std::vector<int64_t> gather_map;
  for (auto idx = range.lo[0]; idx <= range.hi[0]; ++idx) gather_map.push_back(idx - bounds.lo[0]);

  alloc::DeferredBufferAllocator allocator;
  for (auto &pair : args.pairs) {
    auto &in      = input(pair);
    auto &out     = output(pair);
    auto gathered = gather(in.view(), gather_map, false, OutOfRangePolicy::IGNORE, allocator);
    out.return_from_view(allocator, gathered);
  }

  return out_size;
}

void deserialize(Deserializer &ctx, SliceByRangeTask::SliceByRangeTaskArgs &args)
{
  FromFuture<Rect<1>> range;
  deserialize(ctx, range);
  args.range = range.value();

  uint32_t num_values = 0;
  deserialize(ctx, num_values);
  for (uint32_t i = 0; i < num_values; ++i) {
    args.pairs.push_back(SliceByRangeArg{});
    SliceByRangeArg &arg = args.pairs.back();
    deserialize(ctx, input(arg));
    deserialize(ctx, output(arg));
  }

#ifdef DEBUG_PANDAS
  args.sanity_check();
#endif
}

static void __attribute__((constructor)) register_tasks(void)
{
  SliceByRangeTask::register_variants_with_return<int64_t, int64_t>();
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

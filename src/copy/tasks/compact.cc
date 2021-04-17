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

#include "copy/tasks/compact.h"
#include "copy/gather.h"
#include "copy/materialize.h"

namespace legate {
namespace pandas {
namespace copy {

using namespace Legion;

using CompactArg = CompactTask::CompactTaskArgs::CompactArg;

static inline OutputColumn &output(CompactArg &arg) { return arg.first; };

static inline Column<true> &input(CompactArg &arg) { return arg.second; };

void CompactTask::CompactTaskArgs::sanity_check(void)
{
  for (auto &pair : pairs)
    assert(!input(pair).valid() || output(pair).code() == input(pair).code());
  for (auto &pair : pairs) assert(input(pair).shape() == mask.shape());
}

using ColumnView = detail::Column;

namespace detail {

void create_gather_map(Column<true> &mask_col, std::vector<int64_t> &gather_map)
{
  auto mask            = mask_col.raw_column_read<bool>();
  const size_t in_size = mask_col.num_elements();

  if (mask_col.null_count() == 0) {
    for (size_t idx = 0; idx < in_size; ++idx)
      if (mask[idx]) gather_map.push_back(idx);
  } else {
    Bitmask mask_bitmask = mask_col.read_bitmask();
    for (size_t idx = 0; idx < in_size; ++idx)
      if (mask[idx] && mask_bitmask.get(idx)) gather_map.push_back(idx);
  }
}

}  // namespace detail

/*static*/ int64_t CompactTask::cpu_variant(const Task *task,
                                            const std::vector<PhysicalRegion> &regions,
                                            Context context,
                                            Runtime *runtime)
{
  Deserializer ctx{task, regions};

  CompactTaskArgs args;
  deserialize(ctx, args);

  std::vector<int64_t> gather_map;
  detail::create_gather_map(args.mask, gather_map);

  alloc::DeferredBufferAllocator allocator{};
  for (auto &pair : args.pairs) {
    auto &in  = input(pair);
    auto &out = output(pair);

    if (in.valid()) {
      auto gathered = gather(in.view(), gather_map, false, OutOfRangePolicy::IGNORE, allocator);
      out.return_from_view(allocator, gathered);
    } else {
      auto &rect = input(pair).shape();
      auto materialized =
        materialize(rect, args.range_start.value(), args.range_step.value(), allocator);
      auto gathered = gather(materialized, gather_map, false, OutOfRangePolicy::IGNORE, allocator);
      out.return_from_view(allocator, gathered);
    }
  }

  return static_cast<int64_t>(gather_map.size());
}

void deserialize(Deserializer &ctx, CompactTask::CompactTaskArgs &args)
{
  deserialize(ctx, args.mask);

  uint32_t num_values = 0;
  deserialize(ctx, num_values);
  for (uint32_t i = 0; i < num_values; ++i) {
    args.pairs.push_back(CompactArg{});
    CompactArg &arg = args.pairs.back();
    deserialize(ctx, input(arg));
    deserialize(ctx, output(arg));
  }
  Rect<1> in_rect               = args.mask.shape();
  uint32_t num_key_storages     = 0;
  bool input_index_materialized = false;
  deserialize(ctx, num_key_storages);
  deserialize(ctx, input_index_materialized);
  for (uint32_t i = 0; i < num_key_storages; ++i) {
    args.pairs.push_back(CompactArg{});
    CompactArg &arg = args.pairs.back();
    if (input_index_materialized)
      deserialize(ctx, input(arg));
    else {
      deserialize(ctx, args.range_start);
      deserialize(ctx, args.range_step);
      // Still set the right rectangle
      input(arg).set_rect(in_rect);
    }
    deserialize(ctx, output(arg));
  }

  args.sanity_check();
}

static void __attribute__((constructor)) register_tasks(void)
{
  CompactTask::register_variants_with_return<int64_t, int64_t>();
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

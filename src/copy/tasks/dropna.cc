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

#include "copy/tasks/dropna.h"
#include "copy/gather.h"
#include "copy/materialize.h"

namespace legate {
namespace pandas {
namespace copy {

using namespace Legion;

using DropNaArg = DropNaTask::DropNaTaskArgs::DropNaArg;

static inline OutputColumn &output(DropNaArg &arg) { return arg.first; };

static inline Column<true> &input(DropNaArg &arg) { return arg.second; };

void DropNaTask::DropNaTaskArgs::sanity_check(void)
{
  for (auto &pair : pairs)
    assert(!input(pair).valid() || output(pair).code() == input(pair).code());
  for (auto &pair : pairs) assert(input(pair).shape() == input(pairs[0]).shape());
}

void DropNaTask::DropNaTaskArgs::cleanup(void)
{
  for (auto &pair : pairs) {
    output(pair).destroy();
    input(pair).destroy();
  }
}

using ColumnView = detail::Column;

namespace detail {

void create_gather_map(std::vector<Bitmask> &key_bitmasks,
                       uint32_t keep_threshold,
                       std::vector<int64_t> &gather_map)
{
  auto size = key_bitmasks.front().num_elements;

  for (auto idx = 0; idx < size; ++idx) {
    uint32_t num_valids = 0;
    for (auto &mask : key_bitmasks) num_valids += static_cast<uint32_t>(mask.get(idx));
    if (num_valids >= keep_threshold) gather_map.push_back(idx);
  }
}

}  // namespace detail

/*static*/ int64_t DropNaTask::cpu_variant(const Task *task,
                                           const std::vector<PhysicalRegion> &regions,
                                           Context context,
                                           Runtime *runtime)
{
  Deserializer ctx{task, regions};

  DropNaTaskArgs args;
  deserialize(ctx, args);

  std::vector<Bitmask> key_bitmasks;
  for (auto &idx : args.key_indices) key_bitmasks.push_back(input(args.pairs[idx]).read_bitmask());

  std::vector<int64_t> gather_map;
  detail::create_gather_map(key_bitmasks, args.keep_threshold, gather_map);

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

void deserialize(Deserializer &ctx, DropNaTask::DropNaTaskArgs &args)
{
  deserialize(ctx, args.keep_threshold);

  uint32_t num_key_indices = 0;
  deserialize(ctx, num_key_indices);
  args.key_indices.resize(num_key_indices);
  for (auto &idx : args.key_indices) deserialize(ctx, idx);

  uint32_t num_values = 0;
  deserialize(ctx, num_values);
  for (uint32_t i = 0; i < num_values; ++i) {
    args.pairs.push_back(DropNaArg{});
    DropNaArg &arg = args.pairs.back();
    deserialize(ctx, input(arg));
    deserialize(ctx, output(arg));
  }
  Rect<1> in_rect               = input(args.pairs[0]).shape();
  uint32_t num_key_storages     = 0;
  bool input_index_materialized = false;
  deserialize(ctx, num_key_storages);
  deserialize(ctx, input_index_materialized);
  for (uint32_t i = 0; i < num_key_storages; ++i) {
    args.pairs.push_back(DropNaArg{});
    DropNaArg &arg = args.pairs.back();
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
  DropNaTask::register_variants_with_return<int64_t, int64_t>();
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

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

#include <numeric>

#include "sorting/tasks/sort_values.h"
#include "column/detail/column.h"
#include "copy/gather.h"
#include "table/row_wrappers.h"
#include "util/allocator.h"
#include "util/zip_for_each.h"

namespace legate {
namespace pandas {
namespace sorting {

using namespace Legion;

using InputTable     = SortValuesTask::SortValuesArgs::InputTable;
using SortValuesArgs = SortValuesTask::SortValuesArgs;

void SortValuesTask::SortValuesArgs::sanity_check(void)
{
  for (auto &column : input) assert(input[0].shape() == column.shape());
}

/*static*/ int64_t SortValuesTask::cpu_variant(const Task *task,
                                               const std::vector<PhysicalRegion> &regions,
                                               Context context,
                                               Runtime *runtime)
{
  Deserializer ctx{task, regions};

  SortValuesArgs args;
  deserialize(ctx, args);

  std::vector<detail::Column> key_columns;
  for (auto idx : args.key_indices) key_columns.push_back(args.input[idx].view());

  int64_t size = static_cast<int64_t>(args.input[0].num_elements());

  if (size == 0) {
    for (auto &column : args.output) column.make_empty();
    return 0;
  }

  std::vector<int64_t> mapping(size);
  std::iota(mapping.begin(), mapping.end(), 0);

  table::RowCompare op{key_columns, args.ascending, args.put_null_first};
  std::stable_sort(mapping.begin(), mapping.end(), op);

  alloc::DeferredBufferAllocator allocator;

  util::for_each(args.output, args.input, [&](auto &output, auto &input) {
    auto &&gathered =
      copy::gather(input.view(), mapping, false, copy::OutOfRangePolicy::IGNORE, allocator);
    output.return_from_view(allocator, gathered);
  });

  return size;
}

void deserialize(Deserializer &ctx, SortValuesTask::SortValuesArgs &args)
{
  deserialize(ctx, args.use_output_only_columns);
  deserialize(ctx, args.put_null_first);

  uint32_t num_key_columns = 0;
  deserialize(ctx, num_key_columns);
  args.ascending.resize(num_key_columns);
  deserialize(ctx, args.ascending, false);
  args.key_indices.resize(num_key_columns);
  deserialize(ctx, args.key_indices, false);

  uint32_t num_columns = 0;
  deserialize(ctx, num_columns);
  args.input.resize(num_columns);
  deserialize(ctx, args.input, false);
  args.output.resize(num_columns);
  deserialize(ctx, args.output, false);
}

static void __attribute__((constructor)) register_tasks(void)
{
  SortValuesTask::register_variants_with_return<int64_t, int64_t>();
}

}  // namespace sorting
}  // namespace pandas
}  // namespace legate

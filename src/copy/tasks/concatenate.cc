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

#include "copy/tasks/concatenate.h"
#include "copy/concatenate.h"
#include "util/zip_for_each.h"

namespace legate {
namespace pandas {
namespace copy {

using namespace Legion;

void ConcatenateTask::ConcatenateArgs::sanity_check()
{
  for (auto &input_table : input_tables) {
    assert(output_table.size() == input_table.size());
    util::for_each(output_table, input_table, [](const auto &output, const auto &input) {
      assert(output.code() == input.code());
    });
    // FIXME: Here we assume that every input table has at least one column.
    const size_t size = input_table.begin()->num_elements();
    for (auto &column : input_table) assert(column.num_elements() == size);
  }
}

/*static*/ int64_t ConcatenateTask::cpu_variant(const Task *task,
                                                const std::vector<PhysicalRegion> &regions,
                                                Context context,
                                                Runtime *runtime)
{
  Deserializer ctx{task, regions};

  ConcatenateArgs args;
  deserialize(ctx, args);

  int64_t size = 0;
  for (auto &input_table : args.input_tables) size += input_table.begin()->num_elements();

  if (size == 0) {
    for (auto &column : args.output_table) column.allocate(0);
    return 0;
  }

  const auto num_columns = args.output_table.size();
  for (auto col_idx = 0; col_idx < num_columns; ++col_idx) {
    auto &output = args.output_table[col_idx];

    alloc::DeferredBufferAllocator allocator;
    std::vector<detail::Column> inputs;
    for (auto &input_table : args.input_tables) inputs.push_back(input_table[col_idx].view());

    auto &&concatenated = concatenate(inputs, allocator);
    output.return_from_view(allocator, concatenated);
  }
  return size;
}

void deserialize(Deserializer &ctx, ConcatenateTask::ConcatenateArgs &args)
{
  uint32_t num_columns = 0;
  deserialize(ctx, num_columns);
  args.output_table.resize(num_columns);
  for (auto &column : args.output_table) deserialize(ctx, column);

  uint32_t num_inputs = 0;
  deserialize(ctx, num_inputs);
  args.input_tables.resize(num_inputs);
  for (auto &input_table : args.input_tables) {
    input_table.resize(num_columns);
    for (auto &column : input_table) deserialize(ctx, column);
  }
  args.sanity_check();
}

static void __attribute__((constructor)) register_tasks(void)
{
  ConcatenateTask::register_variants_with_return<int64_t, int64_t>();
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

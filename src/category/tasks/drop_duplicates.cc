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

#include <vector>
#include <unordered_set>

#include "category/tasks/drop_duplicates.h"
#include "column/column.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace category {

/*static*/ void DropDuplicatesTask::cpu_variant(const Legion::Task *task,
                                                const std::vector<Legion::PhysicalRegion> &regions,
                                                Legion::Context context,
                                                Legion::Runtime *runtime)
{
  Deserializer ctx{task, regions};

  uint32_t num_inputs = 0;
  deserialize(ctx, num_inputs);

  OutputColumn out;
  deserialize(ctx, out);
  OutputColumn &out_offsets = out.child(0);
  OutputColumn &out_chars   = out.child(1);

  std::vector<Column<true>> inputs;
  for (uint32_t i = 0; i < num_inputs; ++i) {
    Column<true> in;
    deserialize(ctx, in);
    if (!in.valid()) break;
    inputs.push_back(std::move(in));
  }

  std::unordered_set<std::string> distinct_values;
  std::vector<std::unordered_set<std::string>::iterator> result;
  size_t num_chars = 0;

  for (auto &in : inputs) {
    auto in_size = in.num_elements();

    if (in_size == 0) continue;

    auto in_offsets = in.child(0).raw_column_read<int32_t>();
    auto in_chars   = in.child(1).raw_column_read<int8_t>();

    if (in.nullable()) {
      auto in_b = in.read_bitmask();
      for (size_t i = 0; i < in_size; ++i) {
        if (!in_b.get(i)) continue;
        auto lo = in_offsets[i];
        auto hi = in_offsets[i + 1];
        std::string value{&in_chars[lo], &in_chars[hi]};
        auto it = distinct_values.insert(value);
        if (it.second) {
          num_chars += hi - lo;
          result.push_back(it.first);
        }
      }
    } else
      for (size_t i = 0; i < in_size; ++i) {
        auto lo = in_offsets[i];
        auto hi = in_offsets[i + 1];
        std::string value{&in_chars[lo], &in_chars[hi]};
        auto it = distinct_values.insert(value);
        if (it.second) {
          num_chars += hi - lo;
          result.push_back(it.first);
        }
      }
  }

  auto out_size = result.size();

  out.allocate(out_size);
  out_offsets.allocate(out_size > 0 ? out_size + 1 : 0);
  out_chars.allocate(num_chars);

  if (out_size == 0) return;

  auto out_o    = out_offsets.raw_column<int32_t>();
  auto out_c    = out_chars.raw_column<int8_t>();
  size_t out_lo = 0;

  for (size_t i = 0; i < out_size; ++i) {
    auto const &value      = *result[i];
    auto const &value_size = value.size();
    out_o[i]               = out_lo;
    for (size_t k = 0; k < value_size; ++k) out_c[out_lo++] = value[k];
  }
  out_o[out_size] = out_lo;
}

static void __attribute__((constructor)) register_tasks(void)
{
  DropDuplicatesTask::register_variants();
}

}  // namespace category
}  // namespace pandas
}  // namespace legate

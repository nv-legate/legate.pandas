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

#include "string/tasks/strip.h"
#include "column/column.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace string {

using namespace Legion;

/*static*/ void StripTask::cpu_variant(const Task *task,
                                       const std::vector<PhysicalRegion> &regions,
                                       Context context,
                                       Runtime *runtime)
{
  Deserializer ctx{task, regions};

  bool has_to_strip = false;
  std::string to_strip;

  deserialize(ctx, has_to_strip);
  if (has_to_strip)
    deserialize(ctx, to_strip);
  else
    to_strip = " \t\n\r";

  OutputColumn out;
  Column<true> in;
  deserialize(ctx, out);
  deserialize(ctx, in);

  const auto size = in.num_elements();
  if (size == 0) {
    out.make_empty(true);
    return;
  }

  size_t num_chars = 0;
  std::vector<std::string> results;

  auto in_o = in.child(0).raw_column_read<int32_t>();
  auto in_c = in.child(1).raw_column_read<int8_t>();

  for (auto idx = 0; idx < size; ++idx) {
    std::string input(&in_c[in_o[idx]], &in_c[in_o[idx + 1]]);

    auto start_idx = input.find_first_not_of(to_strip);
    auto end_idx   = input.find_last_not_of(to_strip);

    std::string result{input.substr(start_idx, end_idx + 1)};
    results.push_back(result);
    num_chars += result.size();
  }

  out.allocate(size);
  out.child(0).allocate(size + 1);
  out.child(1).allocate(num_chars);

  auto out_o = out.child(0).raw_column<int32_t>();
  auto out_c = out.child(1).raw_column<int8_t>();

  out_o[0] = 0;
  for (auto idx = 0; idx < size; ++idx) {
    out_o[idx + 1] = out_o[idx] + results[idx].size();
    for (auto c : results[idx]) *out_c++ = c;
  }
}

static void __attribute__((constructor)) register_tasks(void) { StripTask::register_variants(); }

}  // namespace string
}  // namespace pandas
}  // namespace legate

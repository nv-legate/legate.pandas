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

#include <regex>

#include "string/tasks/contains.h"
#include "column/column.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace string {

using namespace Legion;

/*static*/ void ContainsTask::cpu_variant(const Task *task,
                                          const std::vector<PhysicalRegion> &regions,
                                          Context context,
                                          Runtime *runtime)
{
  Deserializer ctx{task, regions};

  std::string pattern_string;
  deserialize(ctx, pattern_string);
  std::regex pattern(pattern_string);

  OutputColumn out;
  Column<true> in;
  deserialize(ctx, out);
  deserialize(ctx, in);

  const auto size = in.num_elements();
  if (size == 0) {
    out.make_empty();
    return;
  }

  out.allocate(size);
  auto out_bools = out.raw_column<bool>();

  auto in_offsets = in.child(0).raw_column_read<int32_t>();
  auto in_chars   = in.child(1).raw_column_read<int8_t>();
  for (auto i = 0; i < size; ++i) {
    std::string str{&in_chars[in_offsets[i]], &in_chars[in_offsets[i + 1]]};
    out_bools[i] = std::regex_search(str, pattern);
  }
}

static void __attribute__((constructor)) register_tasks(void) { ContainsTask::register_variants(); }

}  // namespace string
}  // namespace pandas
}  // namespace legate

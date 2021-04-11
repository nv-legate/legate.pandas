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

#include <unordered_map>

#include "category/tasks/encode.h"

namespace legate {
namespace pandas {
namespace category {

/*static*/ void EncodeTask::cpu_variant(const Legion::Task *task,
                                        const std::vector<Legion::PhysicalRegion> &regions,
                                        Legion::Context context,
                                        Legion::Runtime *runtime)
{
  Deserializer ctx{task, regions};

  EncodeTaskArgs args;
  deserialize(ctx, args);

  if (args.in.empty()) {
    args.out.make_empty();
    return;
  }

  auto dict_size    = args.dict.num_elements();
  auto dict_offsets = args.dict.child(0).raw_column_read<int32_t>();
  auto dict_chars   = args.dict.child(1).raw_column_read<int8_t>();

  std::unordered_map<std::string, int32_t> dict;
  dict.reserve(dict_size);
  for (size_t i = 0; i < dict_size; ++i) {
    auto lo = dict_offsets[i];
    auto hi = dict_offsets[i + 1];
    std::string value{&dict_chars[lo], &dict_chars[hi]};
    dict[value] = i;
  }

  const size_t size = args.in.num_elements();
  auto in_offsets   = args.in.child(0).raw_column_read<int32_t>();
  auto in_chars     = args.in.child(1).raw_column_read<int8_t>();

  args.out.allocate(size, true);
  auto out = args.out.child(0).raw_column<uint32_t>();

  if (args.in.nullable()) {
    auto in_b  = args.in.read_bitmask();
    auto out_b = args.out.bitmask();
    for (size_t i = 0; i < size; ++i)
      if (in_b.get(i)) {
        auto lo = in_offsets[i];
        auto hi = in_offsets[i + 1];
        std::string value{&in_chars[lo], &in_chars[hi]};
        auto finder = dict.find(value);
        if (finder != dict.end()) {
          out[i] = dict[value];
          out_b.set(i);
        } else
          out_b.set(i, false);
      } else
        out_b.set(i, false);
  } else if (args.out.nullable()) {
    auto out_b = args.out.bitmask();
    for (size_t i = 0; i < size; ++i) {
      auto lo = in_offsets[i];
      auto hi = in_offsets[i + 1];
      std::string value{&in_chars[lo], &in_chars[hi]};
      auto finder = dict.find(value);
      if (finder != dict.end()) {
        out[i] = dict[value];
        out_b.set(i);
      } else
        out_b.set(i, false);
    }
  } else {
    for (size_t i = 0; i < size; ++i) {
      auto lo = in_offsets[i];
      auto hi = in_offsets[i + 1];
      std::string value{&in_chars[lo], &in_chars[hi]};
#ifdef DEBUG_PANDAS
      assert(dict.find(value) != dict.end());
#endif
      out[i] = dict[value];
    }
  }
}

static void __attribute__((constructor)) register_tasks(void) { EncodeTask::register_variants(); }

}  // namespace category
}  // namespace pandas
}  // namespace legate

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

#include "string/tasks/zfill.h"
#include "column/column.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace string {

using namespace Legion;

/*static*/ void ZfillTask::cpu_variant(const Task *task,
                                       const std::vector<PhysicalRegion> &regions,
                                       Context context,
                                       Runtime *runtime)
{
  Deserializer ctx{task, regions};

  int32_t width;
  deserialize(ctx, width);

  OutputColumn out;
  Column<true> in;
  deserialize(ctx, out);
  deserialize(ctx, in);

  const auto size = in.num_elements();
  if (size == 0) {
    out.make_empty();
    return;
  }

  // First, count the number of characters in the output
  size_t num_chars = 0;
  auto in_offsets  = in.child(0).raw_column_read<int32_t>();
  for (auto i = 0; i < size; ++i) num_chars += std::max(width, in_offsets[i + 1] - in_offsets[i]);

  // Then, we populate the output
  out.allocate(size);
  out.child(0).allocate(size + 1);
  out.child(1).allocate(num_chars);
  auto out_offsets = out.child(0).raw_column<int32_t>();
  auto out_chars   = out.child(1).raw_column<int8_t>();

  out_offsets[0] = 0;
  auto in_chars  = in.child(1).raw_column_read<int8_t>();
  for (auto i = 0; i < size; ++i) {
    auto in_size  = in_offsets[i + 1] - in_offsets[i];
    auto out_size = 0;

    if (in_size >= width) {
      memcpy(out_chars, in_chars, in_size);
      out_size = in_size;
      in_chars += in_size;
      out_chars += out_size;
    } else {
      out_size      = width;
      auto pad_size = width - in_size;
      for (auto k = 0; k < pad_size; ++k) *out_chars++ = '0';
      memcpy(out_chars, in_chars, in_size);
      in_chars += in_size;
      out_chars += in_size;
    }

    out_offsets[i + 1] = out_offsets[i] + out_size;
  }
}

static void __attribute__((constructor)) register_tasks(void) { ZfillTask::register_variants(); }

}  // namespace string
}  // namespace pandas
}  // namespace legate

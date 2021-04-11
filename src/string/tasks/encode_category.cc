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

#include "string/tasks/encode_category.h"
#include "column/column.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace string {

using namespace Legion;

/*static*/ Scalar EncodeCategoryTask::cpu_variant(const Task *task,
                                                  const std::vector<PhysicalRegion> &regions,
                                                  Context context,
                                                  Runtime *runtime)
{
  Deserializer ctx{task, regions};

  Column<true> dictionary;
  Scalar scalar;
  bool can_fail;

  deserialize(ctx, dictionary);
  deserialize(ctx, scalar);
  deserialize(ctx, can_fail);

  if (!scalar.valid()) return Scalar(TypeCode::UINT32);

  auto to_find = scalar.value<std::string>();
  auto size    = dictionary.num_elements();
  auto offsets = dictionary.child(0).raw_column_read<int32_t>();
  auto chars   = dictionary.child(1).raw_column_read<int8_t>();

  for (auto idx = 0; idx < size; ++idx) {
    std::string category(&chars[offsets[idx]], &chars[offsets[idx + 1]]);
    if (category == to_find) return Scalar(true, static_cast<uint32_t>(idx));
  }
  if (!can_fail) {
    fprintf(stderr, "value %s does not exist in the category\n", to_find.c_str());
    assert(false);
  }
  return Scalar(TypeCode::UINT32);
}

static void __attribute__((constructor)) register_tasks(void)
{
  EncodeCategoryTask::register_variants_with_return<Scalar, Scalar>();
}

}  // namespace string
}  // namespace pandas
}  // namespace legate

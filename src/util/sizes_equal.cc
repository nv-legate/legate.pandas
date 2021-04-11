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

#include "util/sizes_equal.h"

namespace legate {
namespace pandas {
namespace util {

using namespace Legion;

/*static*/ Scalar SizesEqualTask::cpu_variant(const Task *task,
                                              const std::vector<PhysicalRegion> &regions,
                                              Context context,
                                              Runtime *runtime)
{
  auto size1 = task->futures[0].get_result<int64_t>();
  auto size2 = task->futures[1].get_result<int64_t>();

  return Scalar(true, size1 == size2);
}

static void __attribute__((constructor)) register_tasks(void)
{
  SizesEqualTask::register_variants_with_return<Scalar, Scalar>();
}

}  // namespace util
}  // namespace pandas
}  // namespace legate

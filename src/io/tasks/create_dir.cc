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

#include <sys/stat.h>

#include "io/tasks/create_dir.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace io {

using namespace Legion;

/*static*/ int32_t CreateDirTask::cpu_variant(const Task *task,
                                              const std::vector<PhysicalRegion> &regions,
                                              Context context,
                                              Runtime *runtime)
{
  Deserializer ctx{task, regions};

  std::string path;
  deserialize(ctx, path);

  auto err = mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  return err;
}

static void __attribute__((constructor)) register_tasks(void)
{
  CreateDirTask::register_variants_with_return<int32_t, int32_t>();
}

}  // namespace io
}  // namespace pandas
}  // namespace legate

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

#include "nccl/tasks/init_nccl_id.h"
#include "nccl/util.h"

namespace legate {
namespace pandas {
namespace util {

/*static*/ ncclUniqueId InitNcclIdTask::gpu_variant(
  const Legion::Task *task,
  const std::vector<Legion::PhysicalRegion> &regions,
  Legion::Context context,
  Legion::Runtime *runtime)
{
  ncclUniqueId id;
  NCCLCHECK(ncclGetUniqueId(&id));
  return id;
}

static void __attribute__((constructor)) register_tasks(void)
{
  InitNcclIdTask::register_variants_with_return<ncclUniqueId, ncclUniqueId>();
}

}  // namespace util
}  // namespace pandas
}  // namespace legate

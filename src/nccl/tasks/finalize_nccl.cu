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

#include "nccl/tasks/finalize_nccl.h"
#include "nccl/util.h"

#include <nccl.h>

namespace legate {
namespace pandas {
namespace util {

/*static*/ void FinalizeNcclTask::gpu_variant(const Legion::Task *task,
                                              const std::vector<Legion::PhysicalRegion> &regions,
                                              Legion::Context context,
                                              Legion::Runtime *runtime)
{
#ifdef DEBUG_PANDAS
  assert(task->futures.size() == 1);
#endif
  auto comm = task->futures[0].get_result<ncclComm_t *>();
  NCCLCHECK(ncclCommDestroy(*comm));
}

static void __attribute__((constructor)) register_tasks(void)
{
  FinalizeNcclTask::register_variants();
}

}  // namespace util
}  // namespace pandas
}  // namespace legate

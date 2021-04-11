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

#include "nccl/tasks/init_nccl.h"
#include "nccl/util.h"
#include "util/gpu_task_context.h"

namespace legate {
namespace pandas {
namespace util {

struct _Payload {
  uint64_t field0;
  uint64_t field1;
};

/*static*/ ncclComm_t *InitNcclTask::gpu_variant(const Legion::Task *task,
                                                 const std::vector<Legion::PhysicalRegion> &regions,
                                                 Legion::Context context,
                                                 Legion::Runtime *runtime)
{
#ifdef DEBUG_PANDAS
  assert(task->futures.size() == 1);
#endif
  auto id          = task->futures[0].get_result<ncclUniqueId>();
  ncclComm_t *comm = new ncclComm_t{};
  NCCLCHECK(ncclGroupStart());
  NCCLCHECK(ncclCommInitRank(comm, task->index_domain.get_volume(), id, task->index_point[0]));
  NCCLCHECK(ncclGroupEnd());

  auto num_pieces = task->index_domain.get_volume();

  if (num_pieces == 1) return comm;

  // Perform a warm-up all-to-all

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  using namespace Legion;

  DeferredBuffer<_Payload, 1> src_buffer(Memory::GPU_FB_MEM,
                                         Domain(Rect<1>{Point<1>{0}, Point<1>{num_pieces - 1}}));

  DeferredBuffer<_Payload, 1> tgt_buffer(Memory::GPU_FB_MEM,
                                         Domain(Rect<1>{Point<1>{0}, Point<1>{num_pieces - 1}}));

  NCCLCHECK(ncclGroupStart());
  for (auto idx = 0; idx < num_pieces; ++idx) {
    NCCLCHECK(ncclSend(src_buffer.ptr(0), sizeof(_Payload), ncclInt8, idx, *comm, stream));
    NCCLCHECK(ncclRecv(tgt_buffer.ptr(0), sizeof(_Payload), ncclInt8, idx, *comm, stream));
  }
  NCCLCHECK(ncclGroupEnd());

  NCCLCHECK(ncclAllGather(src_buffer.ptr(0), tgt_buffer.ptr(0), 1, ncclUint64, *comm, stream));

  return comm;
}

static void __attribute__((constructor)) register_tasks(void)
{
  InitNcclTask::register_variants_with_return<ncclComm_t *, ncclComm_t *>();
}

}  // namespace util
}  // namespace pandas
}  // namespace legate

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

#include "udf/eval_udf.h"
#include "util/cuda_helper.h"
#include "util/gpu_task_context.h"

#include <cuda.h>

namespace legate {
namespace pandas {
namespace udf {

using namespace Legion;

/*static*/ void EvalUDFTask::gpu_variant(const Task *task,
                                         const std::vector<PhysicalRegion> &regions,
                                         Context context,
                                         Runtime *runtime)
{
  Deserializer ctx{task, regions};

  EvalUDFTaskArgs args;
  deserialize(ctx, args);

#ifdef DEBUG_PANDAS
  assert(!args.columns.empty());
#endif
  const auto size = args.columns[0].num_elements();

  args.mask.allocate(size);
  if (size == 0) return;

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  // A technical note: this future is not used when args.scalars was deserialized,
  // as the length of args.scalars passed from the Python side doesn't count this.
  CUfunction func = task->futures.back().get_result<CUfunction>();

  const uint32_t gridDimX = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  const uint32_t gridDimY = 1;
  const uint32_t gridDimZ = 1;

  const uint32_t blockDimX = THREADS_PER_BLOCK;
  const uint32_t blockDimY = 1;
  const uint32_t blockDimZ = 1;

  size_t buffer_size = (args.columns.size() + 1) * sizeof(void *);
  // TODO: We may see alignment issues with the arguments smaller than 4B
  for (auto &scalar : args.scalars) buffer_size += scalar.size_;
  buffer_size += sizeof(size_t);

  std::vector<char> arg_buffer(buffer_size);
  char *raw_arg_buffer = arg_buffer.data();

  auto p                        = raw_arg_buffer;
  *reinterpret_cast<void **>(p) = args.mask.raw_column_untyped();
  p += sizeof(void *);

  for (auto &column : args.columns) {
    *reinterpret_cast<const void **>(p) = column.raw_column_untyped_read();
    p += sizeof(void *);
  }

  for (auto &scalar : args.scalars) {
    memcpy(p, scalar.rawptr_, scalar.size_);
    p += scalar.size_;
  }

  memcpy(p, &size, sizeof(size_t));

  void *config[] = {
    CU_LAUNCH_PARAM_BUFFER_POINTER,
    static_cast<void *>(raw_arg_buffer),
    CU_LAUNCH_PARAM_BUFFER_SIZE,
    &buffer_size,
    CU_LAUNCH_PARAM_END,
  };

  CUresult status = cuLaunchKernel(
    func, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, stream, NULL, config);
  if (status != CUDA_SUCCESS) {
    fprintf(stderr, "Failed to launch a CUDA kernel\n");
    exit(-1);
  }

  if (!args.mask.nullable()) return;

  bool initialized = false;
  Bitmask bitmask  = args.mask.bitmask();
  for (auto &column : args.columns) {
    if (!column.nullable()) continue;
    Bitmask to_merge = column.read_bitmask();
    if (initialized)
      intersect_bitmasks(bitmask, bitmask, to_merge, stream);
    else
      to_merge.copy(bitmask, stream);
  }
}

}  // namespace udf
}  // namespace pandas
}  // namespace legate

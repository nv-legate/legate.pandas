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

#include "copy/tasks/drop_duplicates_nccl.h"
#include "column/column.h"
#include "cudf_util/allocators.h"
#include "cudf_util/column.h"
#include "nccl/shuffle.h"
#include "util/cuda_helper.h"
#include "util/gpu_task_context.h"
#include "deserializer.h"

#include <nccl.h>

#include <cudf/detail/copy.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

namespace legate {
namespace pandas {
namespace copy {

using namespace Legion;

/*static*/ int64_t DropDuplicatesNCCLTask ::gpu_variant(const Task *task,
                                                        const std::vector<PhysicalRegion> &regions,
                                                        Context context,
                                                        Runtime *runtime)
{
  Deserializer ctx{task, regions};

  KeepMethod method;
  deserialize(ctx, method);

  std::vector<int32_t> keys;
  deserialize(ctx, keys);

  uint32_t num_columns = 0;
  deserialize(ctx, num_columns);

  std::vector<Column<true>> inputs;
  std::vector<OutputColumn> outputs;
  inputs.resize(num_columns);
  outputs.resize(num_columns);
  deserialize(ctx, inputs, false);
  deserialize(ctx, outputs, false);

  GPUTaskContext gpu_ctx{};
  auto stream    = gpu_ctx.stream();
  auto num_tasks = static_cast<coord_t>(task->index_domain.get_volume());

  cudf::table_view input_tbl = to_cudf_table(inputs, stream);
  auto keep_option           = to_cudf_keep_option(method);

  DeferredBufferAllocator mr;
  auto temp_mr = rmm::mr::get_current_device_resource();

  auto local_dedup = cudf::detail::drop_duplicates(input_tbl,
                                                   keys,
                                                   keep_option,
                                                   cudf::null_equality::EQUAL,
                                                   stream,
                                                   num_tasks > 1 ? temp_mr : &mr);

  if (num_tasks == 1) {
    auto result_size = static_cast<int64_t>(local_dedup->num_rows());
    from_cudf_table(outputs, std::move(local_dedup), stream, mr);
    return result_size;
  }

  ncclComm_t *comm;
  deserialize_from_future(ctx, comm);

  auto task_id = task->index_point[0];

  auto gathered = comm::all_gather(local_dedup->view(), task_id, num_tasks, comm, stream, temp_mr);

  auto global_dedup = cudf::detail::drop_duplicates(
    gathered->view(), keys, keep_option, cudf::null_equality::EQUAL, stream, temp_mr);

  auto volume    = global_dedup->num_rows();
  auto start_idx = static_cast<cudf::size_type>(volume * task_id / num_tasks);
  auto stop_idx  = static_cast<cudf::size_type>(volume * (task_id + 1) / num_tasks);

  std::vector<cudf::size_type> indices{start_idx, stop_idx};

  auto sliced      = cudf::slice(global_dedup->view(), indices);
  auto result      = std::make_unique<cudf::table>(sliced[0], stream, &mr);
  auto result_size = static_cast<int64_t>(result->num_rows());
  from_cudf_table(outputs, std::move(result), stream, mr);
  return result_size;
}

static void __attribute__((constructor)) register_tasks(void)
{
  DropDuplicatesNCCLTask::register_variants_with_return<int64_t, int64_t>();
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

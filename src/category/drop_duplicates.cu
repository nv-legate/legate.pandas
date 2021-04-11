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

#include "category/drop_duplicates.h"
#include "nccl/shuffle.h"

#include <cudf/detail/stream_compaction.hpp>

namespace legate {
namespace pandas {
namespace category {
namespace detail {

std::unique_ptr<cudf::table> drop_duplicates(const cudf::table_view& input,
                                             std::vector<cudf::size_type> const& keys,
                                             cudf::duplicate_keep_option keep,
                                             cudf::null_equality nulls_equal,
                                             coord_t task_id,
                                             coord_t num_tasks,
                                             ncclComm_t* comm,
                                             cudaStream_t stream,
                                             rmm::mr::device_memory_resource* mr)
{
  auto temp_mr     = rmm::mr::get_current_device_resource();
  auto local_dedup = cudf::detail::drop_duplicates(input, keys, keep, nulls_equal, stream, temp_mr);

  auto concatenated =
    comm::all_gather(local_dedup->view(), task_id, num_tasks, comm, stream, temp_mr);

  return cudf::detail::drop_duplicates(concatenated->view(), keys, keep, nulls_equal, stream, mr);
}

}  // namespace detail
}  // namespace category
}  // namespace pandas
}  // namespace legate

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

#pragma once

#include "pandas.h"

#include <nccl.h>

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

namespace legate {
namespace pandas {
namespace comm {

// Performs distributed all-to-all shuffle using input. The input is first partitioned
// into subtables using splits and they are exchanged so that the i-th task gets the i-th
// pieces from other tasks.
std::unique_ptr<cudf::table> shuffle(const cudf::table_view &input,
                                     const std::vector<cudf::size_type> &splits,
                                     coord_t task_id,
                                     ncclComm_t *comm,
                                     cudaStream_t stream,
                                     rmm::mr::device_memory_resource *mr);

std::unique_ptr<cudf::table> all_gather(const cudf::table_view &input,
                                        coord_t task_id,
                                        coord_t num_tasks,
                                        ncclComm_t *comm,
                                        cudaStream_t stream,
                                        rmm::mr::device_memory_resource *mr);

// Converts all categorical columns to integer columns storing their codes and extracts out
// their dictionaries to a map.
std::pair<cudf::table_view, std::unordered_map<uint32_t, cudf::column_view>> extract_dictionaries(
  const cudf::table_view &input);

// Recovers the original categorical columns using a map of dictionaries that was generated
// by a extract_dictionaries call.
cudf::table_view embed_dictionaries(
  const cudf::table_view &input,
  const std::unordered_map<uint32_t, cudf::column_view> &dictionaries);

}  // namespace comm
}  // namespace pandas
}  // namespace legate

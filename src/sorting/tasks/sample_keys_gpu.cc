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

#include "sorting/tasks/sample_keys.h"
#include "sorting/utilities.h"
#include "column/device_column.h"
#include "util/gpu_task_context.h"
#include "util/zip_for_each.h"

#include <cudf/detail/gather.hpp>

namespace legate {
namespace pandas {
namespace sorting {

using namespace Legion;

using CudfColumns = std::vector<cudf::column_view>;

/*static*/ void SampleKeysTask::gpu_variant(const Task *task,
                                            const std::vector<PhysicalRegion> &regions,
                                            Context context,
                                            Runtime *runtime)
{
  Deserializer ctx{task, regions};

  SampleKeysArgs args;
  deserialize(ctx, args);

  auto size = args.input[0].num_elements();

  if (size == 0) {
    for (auto &column : args.output) column.allocate(0);
    return;
  }

  std::vector<int64_t> mapping;
  sample(size, mapping);
  auto num_samples = mapping.size();

  thrust::device_vector<int64_t> device_mapping{mapping};

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  CudfColumns keys;
  for (auto &column : args.input) keys.push_back(DeviceColumn<true>{column}.to_cudf_column(stream));

  cudf::column_view gather_map{cudf::data_type{cudf::type_id::INT64},
                               static_cast<cudf::size_type>(num_samples),
                               thrust::raw_pointer_cast(&device_mapping[0]),
                               nullptr,
                               0};
  DeferredBufferAllocator mr;
  auto result = cudf::detail::gather(cudf::table_view{std::move(keys)},
                                     gather_map,
                                     cudf::out_of_bounds_policy::DONT_CHECK,
                                     cudf::detail::negative_index_policy::NOT_ALLOWED,
                                     stream,
                                     &mr);
  util::for_each(args.output, result->view(), [&](auto &output, auto &cudf_output) {
    DeviceOutputColumn(output).return_from_cudf_column(mr, cudf_output, stream);
  });
}

}  // namespace sorting
}  // namespace pandas
}  // namespace legate

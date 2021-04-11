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

#include "string/tasks/to_datetime.h"
#include "cudf_util/allocators.h"
#include "util/gpu_task_context.h"
#include "util/zip_for_each.h"
#include "column/column.h"
#include "column/device_column.h"
#include "deserializer.h"

#include <cudf/types.hpp>
#include <cudf/strings/detail/converters.hpp>

namespace legate {
namespace pandas {
namespace string {

using namespace Legion;

/*static*/ void ToDatetimeTask::gpu_variant(const Task *task,
                                            const std::vector<PhysicalRegion> &regions,
                                            Context context,
                                            Runtime *runtime)
{
  Deserializer ctx{task, regions};

  std::string format;
  deserialize(ctx, format);

  OutputColumn h_out;
  Column<true> h_in;
  deserialize(ctx, h_out);
  deserialize(ctx, h_in);

  if (h_in.empty()) {
    h_out.make_empty();
    return;
  }

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  auto in   = DeviceColumn<true>{h_in}.to_cudf_column(stream);
  auto type = cudf::data_type{cudf::type_id::TIMESTAMP_NANOSECONDS};

  DeferredBufferAllocator mr;
  auto result = cudf::strings::detail::to_timestamps(in, type, format, stream, &mr);
  DeviceOutputColumn{h_out}.return_from_cudf_column(mr, result->view(), stream);
}

}  // namespace string
}  // namespace pandas
}  // namespace legate

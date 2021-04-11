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

#include <cctype>

#include "string/tasks/strip.h"
#include "column/column.h"
#include "column/device_column.h"
#include "deserializer.h"

#include "cudf_util/allocators.h"
#include "cudf_util/detail.h"
#include "cudf_util/scalar.h"
#include "util/gpu_task_context.h"

namespace legate {
namespace pandas {
namespace string {

/*static*/ void StripTask::gpu_variant(const Legion::Task *task,
                                       const std::vector<Legion::PhysicalRegion> &regions,
                                       Legion::Context context,
                                       Legion::Runtime *runtime)
{
  Deserializer ctx{task, regions};

  bool has_to_strip = false;
  std::string h_to_strip;

  deserialize(ctx, has_to_strip);
  if (has_to_strip) deserialize(ctx, h_to_strip);

  cudf::string_scalar to_strip(has_to_strip ? h_to_strip : "");

  OutputColumn h_out;
  Column<true> h_in;
  deserialize(ctx, h_out);
  deserialize(ctx, h_in);

  if (h_in.empty()) {
    h_out.make_empty(true);
    return;
  }

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  auto in = DeviceColumn<true>{h_in}.to_cudf_column(stream);

  DeferredBufferAllocator mr;
  auto result =
    cudf::strings::detail::strip(in, cudf::strings::strip_type::BOTH, to_strip, stream, &mr);
  DeviceOutputColumn{h_out}.return_from_cudf_column(mr, result->view(), stream);
}

}  // namespace string
}  // namespace pandas
}  // namespace legate

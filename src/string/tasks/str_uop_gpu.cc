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

#include "string/tasks/str_uop.h"
#include "column/column.h"
#include "column/device_column.h"
#include "deserializer.h"

#include "cudf_util/allocators.h"
#include "cudf_util/detail.h"
#include "util/gpu_task_context.h"

namespace legate {
namespace pandas {
namespace string {

/*static*/ void StrUopTask::gpu_variant(const Legion::Task *task,
                                        const std::vector<Legion::PhysicalRegion> &regions,
                                        Legion::Context context,
                                        Legion::Runtime *runtime)
{
  Deserializer ctx{task, regions};

  int32_t code;
  deserialize(ctx, code);
  const StringMethods op = static_cast<StringMethods>(code);

  OutputColumn out;
  Column<true> in;
  deserialize(ctx, out);
  deserialize(ctx, in);

  if (in.empty()) return;

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  auto num_elements = in.num_elements();
  auto in_column    = DeviceColumn<true>{in}.to_cudf_column(stream);

  DeferredBufferAllocator mr;

  std::unique_ptr<cudf::column> result;
  switch (op) {
    case StringMethods::LOWER: {
      result = cudf::strings::detail::to_lower(in_column, stream, &mr);
      break;
    }
    case StringMethods::UPPER: {
      result = cudf::strings::detail::to_upper(in_column, stream, &mr);
      break;
    }
    case StringMethods::SWAPCASE: {
      result = cudf::strings::detail::swapcase(in_column, stream, &mr);
      break;
    }
    default: {
      assert(false);
      break;
    }
  }
  DeviceOutputColumn{out}.return_from_cudf_column(mr, result->view(), stream);
}

}  // namespace string
}  // namespace pandas
}  // namespace legate

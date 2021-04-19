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

#include "copy/tasks/copy_if_else.h"
#include "column/column.h"
#include "cudf_util/allocators.h"
#include "cudf_util/column.h"
#include "cudf_util/scalar.h"
#include "util/gpu_task_context.h"

#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy.hpp>

namespace legate {
namespace pandas {
namespace copy {

using namespace Legion;

/*static*/ void CopyIfElseTask::gpu_variant(const Task *task,
                                            const std::vector<PhysicalRegion> &regions,
                                            Context context,
                                            Runtime *runtime)
{
  Deserializer ctx{task, regions};

  OutputColumn h_result;
  Column<true> h_input;
  Column<true> h_cond;

  deserialize(ctx, h_result);
  deserialize(ctx, h_input);
  deserialize(ctx, h_cond);

  bool negate{false};
  bool has_other{false};
  bool other_is_scalar{true};
  Column<true> h_other{};
  Scalar other_scalar{};

  deserialize(ctx, negate);
  deserialize(ctx, has_other);
  if (has_other) {
    deserialize(ctx, other_is_scalar);
    if (other_is_scalar)
      deserialize(ctx, other_scalar);
    else
      deserialize(ctx, h_other);
  }

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  DeferredBufferAllocator mr;

  if (has_other && !other_is_scalar) {
    auto input = to_cudf_column(h_input, stream);
    auto cond  = to_cudf_column(h_cond, stream);
    auto other = to_cudf_column(h_other, stream);
    if (negate) {
      auto result = cudf::detail::copy_if_else(other, input, cond, stream, &mr);
      from_cudf_column(h_result, std::move(result), stream, mr);
    } else {
      auto result = cudf::detail::copy_if_else(input, other, cond, stream, &mr);
      from_cudf_column(h_result, std::move(result), stream, mr);
    }
  } else {
    auto input = to_cudf_column(h_input, stream);
    auto cond  = to_cudf_column(h_cond, stream);
    auto scalar =
      to_cudf_scalar(has_other ? other_scalar.raw_ptr() : nullptr, h_input.code(), stream);
    if (negate) {
      auto result = cudf::detail::copy_if_else(*scalar, input, cond, stream, &mr);
      from_cudf_column(h_result, std::move(result), stream, mr);
    } else {
      auto result = cudf::detail::copy_if_else(input, *scalar, cond, stream, &mr);
      from_cudf_column(h_result, std::move(result), stream, mr);
    }
  }
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

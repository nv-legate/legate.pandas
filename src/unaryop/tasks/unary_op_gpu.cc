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

#include "unaryop/tasks/unary_op.h"
#include "column/column.h"
#include "cudf_util/allocators.h"
#include "cudf_util/column.h"
#include "util/gpu_task_context.h"
#include "deserializer.h"

#include <cudf/detail/unary.hpp>

namespace legate {
namespace pandas {
namespace unaryop {

using namespace Legion;

namespace detail {

cudf::unary_operator to_cudf_unary_operator(UnaryOpCode op_code)
{
  switch (op_code) {
    case UnaryOpCode::ABS: {
      return cudf::unary_operator::ABS;
    }
    case UnaryOpCode::BIT_INVERT: {
      return cudf::unary_operator::BIT_INVERT;
    }
  }
  assert(false);
  return cudf::unary_operator::ABS;
}

}  // namespace detail

/*static*/ void UnaryOpTask::gpu_variant(const Task *task,
                                         const std::vector<PhysicalRegion> &regions,
                                         Context context,
                                         Runtime *runtime)
{
  Deserializer ctx{task, regions};

  UnaryOpCode op_code;
  OutputColumn out;
  Column<true> in;

  deserialize(ctx, op_code);
  deserialize(ctx, out);
  deserialize(ctx, in);

  if (in.empty()) {
    out.make_empty();
    return;
  }

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  auto in_col = to_cudf_column(in, stream);

  DeferredBufferAllocator mr;
  auto uop    = detail::to_cudf_unary_operator(op_code);
  auto result = cudf::detail::unary_operation(in_col, uop, stream, &mr);
  from_cudf_column(out, std::move(result), stream, mr);
}

}  // namespace unaryop
}  // namespace pandas
}  // namespace legate

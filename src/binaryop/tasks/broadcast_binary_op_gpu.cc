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

#include "binaryop/tasks/broadcast_binary_op.h"
#include "binaryop/tasks/util.h"
#include "column/column.h"
#include "column/device_column.h"
#include "cudf_util/allocators.h"
#include "cudf_util/scalar.h"
#include "cudf_util/types.h"
#include "util/gpu_task_context.h"
#include "deserializer.h"

#include <cudf/detail/binaryop.hpp>
#include <cudf/detail/replace.hpp>
#include <cudf/scalar/scalar_factories.hpp>

namespace legate {
namespace pandas {
namespace binaryop {

using namespace Legion;

/*static*/ void BroadcastBinaryOpTask::gpu_variant(const Task *task,
                                                   const std::vector<PhysicalRegion> &regions,
                                                   Context context,
                                                   Runtime *runtime)
{
  Deserializer ctx{task, regions};

  BinaryOpCode op_code;
  OutputColumn out;
  Column<true> in;
  Scalar scalar;
  bool scalar_on_rhs;

  deserialize(ctx, op_code);
  deserialize(ctx, out);
  deserialize(ctx, in);
  deserialize(ctx, scalar);
  deserialize(ctx, scalar_on_rhs);

  if (in.empty()) {
    out.make_empty();
    return;
  }

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  auto in1                          = DeviceColumn<true>{in}.to_cudf_column(stream);
  std::unique_ptr<cudf::scalar> in2 = to_cudf_scalar(scalar.raw_ptr(), scalar.code(), stream);

  if (in1.type().id() == cudf::type_id::DICTIONARY32)
    in1 = cudf::column_view(
      in1.child(0).type(), in1.size(), in1.child(0).head(), in1.null_mask(), in1.null_count());

  DeferredBufferAllocator mr;
  auto type_id = cudf::data_type(to_cudf_type_id(out.code()));
  auto binop   = detail::to_cudf_binary_operator(op_code, in.code());

  std::unique_ptr<cudf::column> result;
  if (scalar_on_rhs)
    result = cudf::detail::binary_operation(in1, *in2, binop, type_id, stream, &mr);
  else
    result = cudf::detail::binary_operation(*in2, in1, binop, type_id, stream, &mr);

  // If this is a comparison operator, we need to replace all nulls in the output with false
  if (!out.nullable() && out.code() == TypeCode::BOOL && result->nullable()) {
    auto p_fill_value =
      cudf::make_fixed_width_scalar<bool>(binop == cudf::binary_operator::NOT_EQUAL, stream);
    auto filled = cudf::detail::replace_nulls(result->view(), *p_fill_value, stream, &mr);
    DeviceOutputColumn{out}.return_from_cudf_column(mr, filled->view(), stream);
  } else
    DeviceOutputColumn{out}.return_from_cudf_column(mr, result->view(), stream);
}

}  // namespace binaryop
}  // namespace pandas
}  // namespace legate

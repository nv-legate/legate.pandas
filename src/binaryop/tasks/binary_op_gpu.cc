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

#include "binaryop/tasks/binary_op.h"
#include "binaryop/tasks/util.h"
#include "category/conversion.h"
#include "column/column.h"
#include "column/device_column.h"
#include "cudf_util/allocators.h"
#include "cudf_util/types.h"
#include "cudf_util/scalar.h"
#include "util/gpu_task_context.h"
#include "deserializer.h"

#include <cudf/detail/binaryop.hpp>
#include <cudf/detail/replace.hpp>
#include <cudf/scalar/scalar_factories.hpp>

namespace legate {
namespace pandas {
namespace binaryop {

using namespace Legion;

namespace detail {

Scalar equals(cudf::column_view &&in1, cudf::column_view &&in2, cudaStream_t stream)
{
  auto out = cudf::detail::binary_operation(
    in1, in2, cudf::binary_operator::NULL_EQUALS, cudf::data_type(cudf::type_id::BOOL8), stream);

  DeferredValue<bool> result(true);
  DeferredValue<bool> is_valid(true);

  auto cudf_result = cudf::detail::reduce(
    out->view(), cudf::make_all_aggregation(), cudf::data_type(cudf::type_id::BOOL8), stream);
  return from_cudf_scalar(TypeCode::BOOL, std::move(cudf_result), stream);
}

std::unique_ptr<cudf::column> binop_category(cudf::dictionary_column_view &&in1,
                                             cudf::dictionary_column_view &&in2,
                                             cudf::binary_operator binop,
                                             cudf::data_type type,
                                             cudaStream_t stream,
                                             rmm::mr::device_memory_resource *mr)
{
  auto dicts_equal = equals(in1.keys(), in2.keys(), stream);
  if (dicts_equal.valid() && dicts_equal.value<bool>()) {
    auto codes1 = cudf::column_view(
      in1.indices().type(), in1.size(), in1.indices().head(), in1.null_mask(), in1.null_count());
    auto codes2 = cudf::column_view(
      in2.indices().type(), in2.size(), in2.indices().head(), in2.null_mask(), in2.null_count());
    return cudf::detail::binary_operation(codes1, codes2, binop, type, stream, mr);
  } else {
    // Fall back to a slow path that converts categories to strings for the binary operation
    auto in1_string = category::to_string_column(in1, stream);
    auto in2_string = category::to_string_column(in2, stream);
    return cudf::detail::binary_operation(
      in1_string->view(), in2_string->view(), binop, type, stream, mr);
  }
}

}  // namespace detail

/*static*/ void BinaryOpTask::gpu_variant(const Task *task,
                                          const std::vector<PhysicalRegion> &regions,
                                          Context context,
                                          Runtime *runtime)
{
  Deserializer ctx{task, regions};

  BinaryOpCode op_code;
  OutputColumn out;
  Column<true> in1;
  Column<true> in2;

  deserialize(ctx, op_code);
  deserialize(ctx, out);
  deserialize(ctx, in1);
  deserialize(ctx, in2);

  if (in1.empty()) {
    out.make_empty();
    return;
  }

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  auto in1_col = DeviceColumn<true>{in1}.to_cudf_column(stream);
  auto in2_col = DeviceColumn<true>{in2}.to_cudf_column(stream);

  DeferredBufferAllocator mr;
  auto type_id = cudf::data_type(to_cudf_type_id(out.code()));
  auto binop   = detail::to_cudf_binary_operator(op_code, in1.code());

  std::unique_ptr<cudf::column> result;
  if (in1.code() == TypeCode::CAT32)
    result = detail::binop_category(in1_col, in2_col, binop, type_id, stream, &mr);
  else
    result = cudf::detail::binary_operation(in1_col, in2_col, binop, type_id, stream, &mr);

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

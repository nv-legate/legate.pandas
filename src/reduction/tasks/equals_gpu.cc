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

#include "reduction/tasks/equals.h"
#include "category/conversion.h"
#include "column/column.h"
#include "cudf_util/allocators.h"
#include "cudf_util/column.h"
#include "cudf_util/detail.h"
#include "cudf_util/scalar.h"
#include "util/gpu_task_context.h"
#include "deserializer.h"

#include <cudf/types.hpp>
#include <cudf/detail/binaryop.hpp>
#include <cudf/aggregation.hpp>

namespace legate {
namespace pandas {
namespace reduction {

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

Scalar equals_category(cudf::column_view &&in1, cudf::column_view &&in2, cudaStream_t stream)
{
  auto dicts_equal = equals(in1.child(1), in2.child(1), stream);
  if (dicts_equal.valid() && dicts_equal.value<bool>()) {
    auto codes1 = cudf::column_view(
      in1.child(0).type(), in1.size(), in1.child(0).head(), in1.null_mask(), in1.null_count());
    auto codes2 = cudf::column_view(
      in2.child(0).type(), in2.size(), in2.child(0).head(), in2.null_mask(), in2.null_count());
    return equals(std::move(codes1), std::move(codes2), stream);
  } else {
    // Fall back to a slow path that converts categories to strings for the comparison
    auto in1_string = category::to_string_column(in1, stream);
    auto in2_string = category::to_string_column(in2, stream);
    return equals(in1_string->view(), in2_string->view(), stream);
  }
}

}  // namespace detail

/*static*/ Scalar EqualsTask::gpu_variant(const Task *task,
                                          const std::vector<PhysicalRegion> &regions,
                                          Context context,
                                          Runtime *runtime)
{
  Deserializer ctx{task, regions};

  Column<true> in1;
  Column<true> in2;

  deserialize(ctx, in1);
  deserialize(ctx, in2);

  if (in1.empty()) return Scalar(true, true);
  if (in1.num_elements() != in2.num_elements()) return Scalar(true, false);

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  auto in1_view = to_cudf_column(in1, stream);
  auto in2_view = to_cudf_column(in2, stream);

  if (in1.code() == TypeCode::CAT32)
    return detail::equals_category(std::move(in1_view), std::move(in2_view), stream);
  else
    return detail::equals(std::move(in1_view), std::move(in2_view), stream);
}

}  // namespace reduction
}  // namespace pandas
}  // namespace legate

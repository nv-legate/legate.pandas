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

#include "reduction/tasks/unary_reduction.h"
#include "reduction/reduction_op.h"
#include "column/column.h"
#include "cudf_util/allocators.h"
#include "cudf_util/column.h"
#include "cudf_util/detail.h"
#include "cudf_util/scalar.h"
#include "cudf_util/types.h"
#include "util/gpu_task_context.h"
#include "util/type_dispatch.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace reduction {

using namespace Legion;

namespace detail {

template <AggregationCode AGG>
struct CreateScalar {
  template <TypeCode CODE, std::enable_if_t<is_primitive_type<CODE>::value> * = nullptr>
  Scalar operator()()
  {
    using VAL = pandas_type_of<CODE>;
    return Scalar(true, reduction::Op<AGG, VAL>::identity());
  }
};

template <AggregationCode AGG>
Scalar create_scalar_from_identity(TypeCode type_code)
{
  return type_dispatch_numeric_only(type_code, CreateScalar<AGG>{});
}

}  // namespace detail

/*static*/ Scalar UnaryReductionTask::gpu_variant(const Task *task,
                                                  const std::vector<PhysicalRegion> &regions,
                                                  Context context,
                                                  Runtime *runtime)
{
  Deserializer ctx{task, regions};

  AggregationCode agg_code;
  Column<true> in;
  TypeCode type_code;

  deserialize(ctx, agg_code);
  deserialize(ctx, in);
  deserialize(ctx, type_code);

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  auto in_col = to_cudf_column(in, stream);

  if (agg_code == AggregationCode::COUNT) return Scalar(true, in_col.size() - in_col.null_count());

  DeferredBufferAllocator mr;
  auto agg     = to_cudf_agg(agg_code);
  auto type_id = cudf::data_type{to_cudf_type_id(type_code)};
  auto out     = cudf::detail::reduce(in_col, std::move(agg), type_id, stream, &mr);

  if (out->is_valid())
    return from_cudf_scalar(type_code, std::move(out), stream);
  else {
    switch (agg_code) {
      case AggregationCode::SUM: {
        return detail::create_scalar_from_identity<AggregationCode::SUM>(type_code);
      }
      case AggregationCode::PROD: {
        return detail::create_scalar_from_identity<AggregationCode::PROD>(type_code);
      }
      default: {
        return Scalar(type_code);
      }
    }
  }
}

}  // namespace reduction
}  // namespace pandas
}  // namespace legate

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

#include "reduction/tasks/scalar_reduction.h"
#include "reduction/reduction_op.h"
#include "column/column.h"
#include "scalar/scalar.h"
#include "util/type_dispatch.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace reduction {

using namespace Legion;

namespace detail {

template <TypeCode CODE>
struct ReductionImpl {
  template <AggregationCode AGG,
            std::enable_if_t<(reduction::is_numeric_aggregation<AGG>::value ||
                              AGG == AggregationCode::MIN || AGG == AggregationCode::MAX) &&
                             !reduction::is_compound_aggregation<AGG>::value> * = nullptr>
  Scalar operator()(const std::vector<Scalar> &inputs)
  {
    using VAL     = pandas_type_of<CODE>;
    using OP_TYPE = reduction::Op<AGG, VAL>;
    using RES     = typename OP_TYPE::result_t;

    OP_TYPE op{};

    Scalar out(false, OP_TYPE::identity());

    for (auto &input : inputs) {
      if (!input.valid()) continue;
      out.value<RES>() = op(input.value<VAL>(), out.value<RES>());
      out.set_valid(true);
    }

    // Here come ugly hacks to match the semantics with vanilla Pandas;
    // since vanilla Pandas doesn't have a nullable scalar, it returns
    // the identity value when the result is invalid and the reduction
    // was either a sum or product.
    if (!out.valid() && !(AGG == AggregationCode::MAX || AGG == AggregationCode::MIN)) {
      out.value<RES>() = OP_TYPE::identity();
      out.set_valid(true);
    }

    return out;
  }

  template <AggregationCode AGG, std::enable_if_t<AGG == AggregationCode::MEAN> * = nullptr>
  Scalar operator()(const std::vector<Scalar> &inputs)
  {
#ifdef DEBUG_PANDAS
    assert(inputs.size() == 2);
#endif
    using VAL = pandas_type_of<CODE>;

    auto sum = inputs[0].value<VAL>();
    auto cnt = inputs[1].value<int32_t>();

    auto valid = inputs[0].valid() && inputs[1].valid() && cnt > 0;
    double value{cnt > 0 ? static_cast<double>(sum) / static_cast<double>(cnt) : std::nan("")};

    return Scalar(valid, value);
  }

  template <AggregationCode AGG, std::enable_if_t<AGG == AggregationCode::VAR> * = nullptr>
  Scalar operator()(const std::vector<Scalar> &inputs)
  {
#ifdef DEBUG_PANDAS
    assert(inputs.size() == 3);
#endif
    using VAL = pandas_type_of<CODE>;

    auto cnt = inputs[2].value<int32_t>();

    double value{std::nan("")};
    bool valid{false};
    if (cnt > 1) {
      auto cnt_casted = static_cast<double>(cnt);
      auto sqmean     = static_cast<double>(inputs[0].value<VAL>()) / cnt_casted;
      auto mean       = static_cast<double>(inputs[1].value<VAL>()) / cnt_casted;
      value           = cnt_casted / (cnt_casted - 1) * (sqmean - mean * mean);
      valid           = true;
    }

    return Scalar(valid, value);
  }

  template <AggregationCode AGG, std::enable_if_t<AGG == AggregationCode::STD> * = nullptr>
  Scalar operator()(const std::vector<Scalar> &inputs)
  {
    auto var = operator()<AggregationCode::VAR>(inputs);

    return Scalar(var.valid(), std::sqrt(var.template value<double>()));
  }

  template <AggregationCode AGG, std::enable_if_t<AGG == AggregationCode::COUNT> * = nullptr>
  Scalar operator()(const std::vector<Scalar> &inputs)
  {
    assert(false);
    return Scalar();
  }

  template <AggregationCode AGG, std::enable_if_t<AGG == AggregationCode::SIZE> * = nullptr>
  Scalar operator()(const std::vector<Scalar> &inputs)
  {
    assert(false);
    return Scalar();
  }
};

struct StringReductionImpl {
  template <
    AggregationCode AGG,
    std::enable_if_t<AGG == AggregationCode::MIN || AGG == AggregationCode::MAX> * = nullptr>
  Scalar operator()(const std::vector<Scalar> &inputs)
  {
    using VAL     = std::string;
    using OP_TYPE = reduction::Op<AGG, VAL>;
    using RES     = std::string;

    OP_TYPE op{};

    bool valid = false;
    std::string result;

    for (auto &input : inputs) {
      if (!input.valid()) continue;
      auto &value = input.value<std::string>();
      if (valid)
        result = op(result, value);
      else {
        valid  = true;
        result = value;
      }
    }

    return valid ? Scalar(valid, result) : Scalar(TypeCode::STRING);
  }

  template <
    AggregationCode AGG,
    std::enable_if_t<!(AGG == AggregationCode::MIN || AGG == AggregationCode::MAX)> * = nullptr>
  Scalar operator()(const std::vector<Scalar> &inputs)
  {
    assert(false);
    return Scalar();
  }
};

struct TypeDispatch {
  template <TypeCode CODE, std::enable_if_t<is_primitive_type<CODE>::value> * = nullptr>
  Scalar operator()(AggregationCode agg_code, const std::vector<Scalar> &inputs)
  {
    return type_dispatch(agg_code, ReductionImpl<CODE>{}, inputs);
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::STRING> * = nullptr>
  Scalar operator()(AggregationCode agg_code, const std::vector<Scalar> &inputs)
  {
    return type_dispatch(agg_code, StringReductionImpl{}, inputs);
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::CAT32> * = nullptr>
  Scalar operator()(AggregationCode agg_code, const std::vector<Scalar> &inputs)
  {
    assert(false);
    return Scalar();
  }
};

Scalar scalar_reduction(AggregationCode agg_code, const std::vector<Scalar> &inputs)
{
  return type_dispatch(inputs.front().code(), TypeDispatch{}, agg_code, inputs);
}

}  // namespace detail

/*static*/ Scalar ScalarReductionTask::cpu_variant(const Task *task,
                                                   const std::vector<PhysicalRegion> &regions,
                                                   Context context,
                                                   Runtime *runtime)
{
  Deserializer ctx{task, regions};

  AggregationCode agg_code;
  std::vector<Scalar> inputs;

  deserialize(ctx, agg_code);
  for (auto &future : task->futures) inputs.push_back(future.get_result<Scalar>());

  return detail::scalar_reduction(agg_code, inputs);
}

static void __attribute__((constructor)) register_tasks(void)
{
  ScalarReductionTask::register_variants_with_return<Scalar, Scalar>();
}

}  // namespace reduction
}  // namespace pandas
}  // namespace legate

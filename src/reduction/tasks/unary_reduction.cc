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

#include <cmath>

#include "reduction/tasks/unary_reduction.h"
#include "reduction/reduction_op.h"
#include "column/column.h"
#include "util/type_dispatch.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace reduction {

using namespace Legion;
using ColumnView = detail::Column;

namespace detail {

template <TypeCode CODE>
struct ReductionImpl {
  template <AggregationCode AGG,
            std::enable_if_t<(reduction::is_numeric_aggregation<AGG>::value ||
                              AGG == AggregationCode::MIN || AGG == AggregationCode::MAX) &&
                             !reduction::is_compound_aggregation<AGG>::value> * = nullptr>
  Scalar operator()(const ColumnView &in_value)
  {
    using VAL     = pandas_type_of<CODE>;
    using OP_TYPE = reduction::Op<AGG, VAL>;

    OP_TYPE op{};

    auto in            = in_value.column<VAL>();
    const auto in_size = in_value.size();

    Scalar out(false, OP_TYPE::identity());

    if (in_value.nullable()) {
      auto in_b = in_value.bitmask();
      for (size_t i = 0; i < in_size; ++i) {
        if (!in_b.get(i)) continue;
        out.value<VAL>() = op(out.value<VAL>(), in[i]);
        out.set_valid(true);
      }
    } else {
      for (size_t i = 0; i < in_size; ++i) {
        out.value<VAL>() = op(out.value<VAL>(), in[i]);
        out.set_valid(true);
      }
    }

    // Here come ugly hacks to match the semantics with vanilla Pandas;
    // since vanilla Pandas doesn't have a nullable scalar, it returns
    // the identity value when the result is invalid and the reduction
    // was one of sum, product, any, and all.
    if (!out.valid() && !(AGG == AggregationCode::MAX || AGG == AggregationCode::MIN)) {
      out.value<VAL>() = OP_TYPE::identity();
      out.set_valid(true);
    }

    return out;
  }

  template <AggregationCode AGG, std::enable_if_t<AGG == AggregationCode::MEAN> * = nullptr>
  Scalar operator()(const ColumnView &in_value)
  {
    using VAL = pandas_type_of<CODE>;

    auto sum = operator()<AggregationCode::SUM>(in_value);
    auto cnt = operator()<AggregationCode::COUNT>(in_value);

    auto cnt_casted = static_cast<double>(cnt.template value<int32_t>());
    auto mean       = static_cast<double>(sum.template value<VAL>()) / cnt_casted;

    return Scalar(cnt.template value<int32_t>() > 0, mean);
  }

  template <AggregationCode AGG, std::enable_if_t<AGG == AggregationCode::VAR> * = nullptr>
  Scalar operator()(const ColumnView &in_value)
  {
    using VAL = pandas_type_of<CODE>;

    auto sqsum = operator()<AggregationCode::SQSUM>(in_value);
    auto sum   = operator()<AggregationCode::SUM>(in_value);
    auto cnt   = operator()<AggregationCode::COUNT>(in_value);

    auto cnt_casted = static_cast<double>(cnt.template value<int32_t>());
    auto sqmean     = static_cast<double>(sqsum.template value<VAL>()) / cnt_casted;
    auto mean       = static_cast<double>(sum.template value<VAL>()) / cnt_casted;
    auto var        = cnt_casted / (cnt_casted - 1) * (sqmean - mean * mean);

    return Scalar(cnt.template value<int32_t>() > 1, var);
  }

  template <AggregationCode AGG, std::enable_if_t<AGG == AggregationCode::STD> * = nullptr>
  Scalar operator()(const ColumnView &in_value)
  {
    auto var = operator()<AggregationCode::VAR>(in_value);
    return Scalar(var.valid(), std::sqrt(var.template value<double>()));
  }

  // ----------------------------------------------------------------------------------------

  template <AggregationCode AGG, std::enable_if_t<AGG == AggregationCode::COUNT> * = nullptr>
  Scalar operator()(const ColumnView &in_value)
  {
    if (in_value.nullable()) {
      const auto in_size = in_value.size();
      auto in_b          = in_value.bitmask();
      auto null_count    = in_b.count_unset_bits();
      return Scalar(true, static_cast<int32_t>(in_size - null_count));
    } else
      return Scalar(true, static_cast<int32_t>(in_value.size()));
  }

  template <AggregationCode AGG, std::enable_if_t<AGG == AggregationCode::SIZE> * = nullptr>
  Scalar operator()(const ColumnView &in_value)
  {
    assert(false);
    return Scalar();
  }
};

struct StringReductionImpl {
  template <
    AggregationCode AGG,
    std::enable_if_t<AGG == AggregationCode::MIN || AGG == AggregationCode::MAX> * = nullptr>
  Scalar operator()(const ColumnView &in_value)
  {
    using OP_TYPE = reduction::Op<AGG, std::string>;

    OP_TYPE op{};

    bool initialized = false;
    std::string out;
    const auto in_size = in_value.size();

    if (in_value.nullable()) {
      auto in_b = in_value.bitmask();
      for (size_t i = 0; i < in_size; ++i) {
        if (!in_b.get(i)) continue;

        auto in = in_value.element<std::string>(i);
        if (!initialized) {
          initialized = true;
          out         = in;
        } else
          out = op(out, in);
      }
    } else {
      for (size_t i = 0; i < in_size; ++i) {
        auto in = in_value.element<std::string>(i);
        if (!initialized) {
          initialized = true;
          out         = in;
        } else
          out = op(out, in);
      }
    }

    return initialized ? Scalar(initialized, out) : Scalar(TypeCode::STRING);
  }

  template <AggregationCode AGG, std::enable_if_t<AGG == AggregationCode::COUNT> * = nullptr>
  Scalar operator()(const ColumnView &in_value)
  {
    if (in_value.nullable()) {
      const auto in_size = in_value.size();
      auto in_b          = in_value.bitmask();
      auto null_count    = in_b.count_unset_bits();
      return Scalar(true, static_cast<int32_t>(in_size - null_count));
    } else
      return Scalar(true, static_cast<int32_t>(in_value.size()));
  }

  template <AggregationCode AGG,
            std::enable_if_t<!(AGG == AggregationCode::COUNT || AGG == AggregationCode::MIN ||
                               AGG == AggregationCode::MAX)> * = nullptr>
  Scalar operator()(const ColumnView &in_value)
  {
    assert(false);
    return Scalar();
  }
};

struct TypeDispatch {
  template <TypeCode CODE, std::enable_if_t<is_primitive_type<CODE>::value> * = nullptr>
  Scalar operator()(AggregationCode agg_code, const ColumnView &in_value)
  {
    return type_dispatch(agg_code, ReductionImpl<CODE>{}, in_value);
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::STRING> * = nullptr>
  Scalar operator()(AggregationCode agg_code, const ColumnView &in_value)
  {
    return type_dispatch(agg_code, StringReductionImpl{}, in_value);
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::CAT32> * = nullptr>
  Scalar operator()(AggregationCode agg_code, const ColumnView &in_value)
  {
    assert(false);
    return Scalar(false, 0);
  }
};

Scalar unary_reduction(AggregationCode agg_code, const ColumnView &in_value)
{
  return type_dispatch(in_value.code(), TypeDispatch{}, agg_code, in_value);
}

}  // namespace detail

/*static*/ Scalar UnaryReductionTask::cpu_variant(const Task *task,
                                                  const std::vector<PhysicalRegion> &regions,
                                                  Context context,
                                                  Runtime *runtime)
{
  Deserializer ctx{task, regions};

  AggregationCode agg_code;
  Column<true> in;

  deserialize(ctx, agg_code);
  deserialize(ctx, in);

  return detail::unary_reduction(agg_code, in.view());
}

static void __attribute__((constructor)) register_tasks(void)
{
  UnaryReductionTask::register_variants_with_return<Scalar, Scalar>();
}

}  // namespace reduction
}  // namespace pandas
}  // namespace legate

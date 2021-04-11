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

#include "reduction/tasks/equals.h"
#include "category/utilities.h"
#include "column/column.h"
#include "util/type_dispatch.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace reduction {

using namespace Legion;
using ColumnView = detail::Column;

namespace detail {

// TODO: This code should be simplified as an elementwise binary equal followed by an all reduction.

struct Equals {
  template <TypeCode CODE, std::enable_if_t<is_primitive_type<CODE>::value> * = nullptr>
  Scalar operator()(const ColumnView &in1, const ColumnView &in2)
  {
    using VAL = pandas_type_of<CODE>;

#ifdef DEBUG_PANDAS
    assert(in1.size() == in2.size());
#endif

    auto size  = in1.size();
    auto p_in1 = in1.column<VAL>();
    auto p_in2 = in2.column<VAL>();

    if (!in1.nullable() && !in2.nullable()) {
      for (size_t idx = 0; idx < size; ++idx)
        if (p_in1[idx] != p_in2[idx]) return Scalar(true, false);
    } else if (in1.nullable() && in2.nullable()) {
      auto in1_b = in1.bitmask();
      auto in2_b = in2.bitmask();
      for (size_t idx = 0; idx < size; ++idx) {
        if (in1_b.get(idx) != in2_b.get(idx))
          return Scalar(true, false);
        else if (in1_b.get(idx) && p_in1[idx] != p_in2[idx])
          return Scalar(true, false);
      }
    } else {
      auto in_b = in1.nullable() ? in1.bitmask() : in2.bitmask();
      for (size_t idx = 0; idx < size; ++idx)
        if (!in_b.get(idx) || p_in1[idx] != p_in2[idx]) return Scalar(true, false);
    }

    return Scalar(true, true);
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::STRING> * = nullptr>
  Scalar operator()(const ColumnView &in1, const ColumnView &in2)
  {
#ifdef DEBUG_PANDAS
    assert(in1.size() == in2.size());
#endif

    auto size = in1.size();

    auto p_in1_o = in1.child(0).template column<int32_t>();
    auto p_in1_c = in1.child(1).template column<int8_t>();

    auto p_in2_o = in2.child(0).template column<int32_t>();
    auto p_in2_c = in2.child(1).template column<int8_t>();

    if (!in1.nullable() && !in2.nullable()) {
      for (size_t idx = 0; idx < size; ++idx) {
        std::string lh(&p_in1_c[p_in1_o[idx]], &p_in1_c[p_in1_o[idx + 1]]);
        std::string rh(&p_in2_c[p_in2_o[idx]], &p_in2_c[p_in2_o[idx + 1]]);
        if (lh != rh) return Scalar(true, false);
      }
    } else if (in1.nullable() && in2.nullable()) {
      auto in1_b = in1.bitmask();
      auto in2_b = in2.bitmask();
      for (size_t idx = 0; idx < size; ++idx) {
        if (in1_b.get(idx) != in2_b.get(idx)) return Scalar(true, false);
        if (!in1_b.get(idx)) continue;
        std::string lh(&p_in1_c[p_in1_o[idx]], &p_in1_c[p_in1_o[idx + 1]]);
        std::string rh(&p_in2_c[p_in2_o[idx]], &p_in2_c[p_in2_o[idx + 1]]);
        if (lh != rh) return Scalar(true, false);
      }
    } else {
      auto in_b = in1.nullable() ? in1.bitmask() : in2.bitmask();
      for (size_t idx = 0; idx < size; ++idx) {
        if (!in_b.get(idx)) return Scalar(true, false);
        std::string lh(&p_in1_c[p_in1_o[idx]], &p_in1_c[p_in1_o[idx + 1]]);
        std::string rh(&p_in2_c[p_in2_o[idx]], &p_in2_c[p_in2_o[idx + 1]]);
        if (lh != rh) return Scalar(true, false);
      }
    }

    return Scalar(true, true);
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::CAT32> * = nullptr>
  Scalar operator()(const ColumnView &in1, const ColumnView &in2)
  {
    std::vector<std::string> dict1, dict2;

    // TODO: We should change to_dictionary to return a vector and move it.
    category::to_dictionary(dict1, in1.child(1));
    category::to_dictionary(dict2, in2.child(1));

    auto size  = in1.size();
    auto p_in1 = in1.child(0).column<uint32_t>();
    auto p_in2 = in2.child(0).column<uint32_t>();

    if (!in1.nullable() && !in2.nullable()) {
      for (size_t idx = 0; idx < size; ++idx)
        if (dict1[p_in1[idx]] != dict2[p_in2[idx]]) return Scalar(true, false);
    } else if (in1.nullable() && in2.nullable()) {
      auto in1_b = in1.bitmask();
      auto in2_b = in2.bitmask();
      for (size_t idx = 0; idx < size; ++idx) {
        if (in1_b.get(idx) != in2_b.get(idx))
          return Scalar(true, false);
        else if (in1_b.get(idx) && dict1[p_in1[idx]] != dict2[p_in2[idx]])
          return Scalar(true, false);
      }
    } else {
      auto in_b = in1.nullable() ? in1.bitmask() : in2.bitmask();
      for (size_t idx = 0; idx < size; ++idx)
        if (!in_b.get(idx) || dict1[p_in1[idx]] != dict2[p_in2[idx]]) return Scalar(true, false);
    }

    return Scalar(true, true);
  }
};

Scalar equals(const ColumnView &in1, const ColumnView &in2)
{
  if (in1.size() != in2.size()) return Scalar(true, false);
  return type_dispatch(in1.code(), Equals{}, in1, in2);
}

}  // namespace detail

/*static*/ Scalar EqualsTask::cpu_variant(const Task *task,
                                          const std::vector<PhysicalRegion> &regions,
                                          Context context,
                                          Runtime *runtime)
{
  Deserializer ctx{task, regions};

  Column<true> in1;
  Column<true> in2;

  deserialize(ctx, in1);
  deserialize(ctx, in2);

  return detail::equals(in1.view(), in2.view());
}

static void __attribute__((constructor)) register_tasks(void)
{
  EqualsTask::register_variants_with_return<Scalar, Scalar>();
}

}  // namespace reduction
}  // namespace pandas
}  // namespace legate

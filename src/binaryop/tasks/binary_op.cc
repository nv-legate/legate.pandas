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
#include "binaryop/binary_op.h"
#include "category/utilities.h"
#include "column/column.h"
#include "util/allocator.h"
#include "util/type_dispatch.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace binaryop {

using namespace Legion;
using ColumnView = detail::Column;

namespace detail {

namespace numeric {

template <BinaryOpCode OP_CODE>
struct ArithmeticBinaryOpImpl {
  template <TypeCode TYPE_CODE,
            std::enable_if_t<is_numeric_type<TYPE_CODE>::value && is_arithmetic_op<OP_CODE>::value>
              * = nullptr>
  ColumnView operator()(const ColumnView &in1, const ColumnView &in2, alloc::Allocator &allocator)
  {
    using ARG = pandas_type_of<TYPE_CODE>;
    using OP  = BinaryOp<OP_CODE, ARG>;
    using RES = std::result_of_t<OP(ARG, ARG)>;

    OP op{};
    auto res_code = pandas_type_code_of<RES>;

    auto size  = in1.size();
    auto out   = allocator.allocate_elements<RES>(size);
    auto p_in1 = in1.column<ARG>();
    auto p_in2 = in2.column<ARG>();

    if (!in1.nullable() && !in2.nullable()) {
      for (auto idx = 0; idx < size; ++idx) out[idx] = op(p_in1[idx], p_in2[idx]);
      return ColumnView(res_code, out, size);
    } else if (in1.nullable() || in2.nullable()) {
      Bitmask in_b(size, allocator);
      in_b.set_all_valid();
      if (in1.nullable()) intersect_bitmasks(in_b, in_b, in1.bitmask());
      if (in2.nullable()) intersect_bitmasks(in_b, in_b, in2.bitmask());

      Bitmask out_b(size, allocator);
      in_b.copy(out_b);

      for (auto idx = 0; idx < size; ++idx) {
        bool valid = in_b.get(idx);
        if (valid) out[idx] = op(p_in1[idx], p_in2[idx]);
      }
      return ColumnView(res_code, out, size, out_b.raw_ptr());
    }
    assert(false);
    return ColumnView();
  }

  template <TypeCode TYPE_CODE,
            std::enable_if_t<!is_numeric_type<TYPE_CODE>::value ||
                             !is_arithmetic_op<OP_CODE>::value> * = nullptr>
  ColumnView operator()(const ColumnView &in1, const ColumnView &in2, alloc::Allocator &allocator)
  {
    assert(false);
    return ColumnView(TYPE_CODE);
  }
};

template <BinaryOpCode OP_CODE>
struct ComparisonBinaryOpImpl {
  template <TypeCode TYPE_CODE,
            std::enable_if_t<is_primitive_type<TYPE_CODE>::value &&
                             !is_arithmetic_op<OP_CODE>::value> * = nullptr>
  ColumnView operator()(const ColumnView &in1, const ColumnView &in2, alloc::Allocator &allocator)
  {
    using ARG = pandas_type_of<TYPE_CODE>;
    using OP  = BinaryOp<OP_CODE, ARG>;
    using RES = std::result_of_t<OP(ARG, ARG)>;

    OP op{};
    auto res_code = pandas_type_code_of<RES>;

    auto size  = in1.size();
    auto out   = allocator.allocate_elements<RES>(size);
    auto p_in1 = in1.column<ARG>();
    auto p_in2 = in2.column<ARG>();

    if (!in1.nullable() && !in2.nullable()) {
      for (auto idx = 0; idx < size; ++idx) out[idx] = op(p_in1[idx], p_in2[idx]);
      return ColumnView(res_code, out, size);
    } else if (in1.nullable() || in2.nullable()) {
      Bitmask in_b(size, allocator);
      in_b.set_all_valid();
      if (in1.nullable()) intersect_bitmasks(in_b, in_b, in1.bitmask());
      if (in2.nullable()) intersect_bitmasks(in_b, in_b, in2.bitmask());

      for (auto idx = 0; idx < size; ++idx) {
        bool valid = in_b.get(idx);
        if (valid)
          out[idx] = op(p_in1[idx], p_in2[idx]);
        else
          out[idx] = valid != (OP_CODE == BinaryOpCode::NOT_EQUAL);
      }
      return ColumnView{res_code, out, size};
    }
    assert(false);
    return ColumnView();
  }

  template <TypeCode TYPE_CODE,
            std::enable_if_t<!is_primitive_type<TYPE_CODE>::value ||
                             is_arithmetic_op<OP_CODE>::value> * = nullptr>
  ColumnView operator()(const ColumnView &in1, const ColumnView &in2, alloc::Allocator &allocator)
  {
    assert(false);
    return ColumnView(TYPE_CODE);
  }
};

struct NumericBinaryOpImpl {
  template <BinaryOpCode OP>
  ColumnView operator()(const ColumnView &in1, const ColumnView &in2, alloc::Allocator &allocator)
  {
    if (is_arithmetic_op<OP>::value)
      return type_dispatch(in1.code(), ArithmeticBinaryOpImpl<OP>{}, in1, in2, allocator);
    else
      return type_dispatch(in1.code(), ComparisonBinaryOpImpl<OP>{}, in1, in2, allocator);
  }
};

ColumnView binary_op(BinaryOpCode op_code,
                     const ColumnView &in1,
                     const ColumnView &in2,
                     alloc::Allocator &allocator)
{
  return type_dispatch(op_code, NumericBinaryOpImpl{}, in1, in2, allocator);
}

}  // namespace numeric

namespace string {

struct StringBinaryOpImpl {
  template <BinaryOpCode OP_CODE, std::enable_if_t<!is_arithmetic_op<OP_CODE>::value> * = nullptr>
  ColumnView operator()(const ColumnView &in1, const ColumnView &in2, alloc::Allocator &allocator)
  {
    using ARG = std::string;
    using OP  = BinaryOp<OP_CODE, ARG>;
    using RES = std::result_of_t<OP(ARG, ARG)>;

    OP op{};
    auto res_code = pandas_type_code_of<RES>;

    auto size = in1.size();
    auto out  = allocator.allocate_elements<RES>(size);

    auto in1_o = in1.child(0).column<int32_t>();
    auto in2_o = in2.child(0).column<int32_t>();

    auto in1_c = in1.child(1).column<int8_t>();
    auto in2_c = in2.child(1).column<int8_t>();

    if (!in1.nullable() && !in2.nullable()) {
      for (auto idx = 0; idx < size; ++idx) {
        std::string v1(&in1_c[in1_o[idx]], &in1_c[in1_o[idx + 1]]);
        std::string v2(&in2_c[in2_o[idx]], &in2_c[in2_o[idx + 1]]);
        out[idx] = op(v1, v2);
      }
      return ColumnView(res_code, out, size);
    } else if (in1.nullable() || in2.nullable()) {
      Bitmask in_b(size, allocator);
      in_b.set_all_valid();
      if (in1.nullable()) intersect_bitmasks(in_b, in_b, in1.bitmask());
      if (in2.nullable()) intersect_bitmasks(in_b, in_b, in2.bitmask());

      for (auto idx = 0; idx < size; ++idx) {
        bool valid = in_b.get(idx);
        if (valid) {
          std::string v1(&in1_c[in1_o[idx]], &in1_c[in1_o[idx + 1]]);
          std::string v2(&in2_c[in2_o[idx]], &in2_c[in2_o[idx + 1]]);
          out[idx] = op(v1, v2);
        } else
          out[idx] = valid != (OP_CODE == BinaryOpCode::NOT_EQUAL);
      }
      return ColumnView{res_code, out, size};
    }
    assert(false);
    return ColumnView();
  }

  template <BinaryOpCode OP_CODE, std::enable_if_t<is_arithmetic_op<OP_CODE>::value> * = nullptr>
  ColumnView operator()(const ColumnView &in1, const ColumnView &in2, alloc::Allocator &allocator)
  {
    assert(false);
    return ColumnView{};
  }
};

ColumnView binary_op(BinaryOpCode op_code,
                     const ColumnView &in1,
                     const ColumnView &in2,
                     alloc::Allocator &allocator)
{
  return type_dispatch(op_code, StringBinaryOpImpl{}, in1, in2, allocator);
}

}  // namespace string

namespace category {

struct DictEqual {
  DictEqual(const std::vector<std::string> &dict1, const std::vector<std::string> &dict2)
    : dict1_(dict1), dict2_(dict2)
  {
  }

  bool operator()(const uint32_t &a, const uint32_t &b) const
  {
#ifdef DEBUG_PANDAS
    assert(0 <= a && a < dict1_.size());
    assert(0 <= b && b < dict2_.size());
#endif
    return dict1_[a] == dict2_[b];
  }

  const std::vector<std::string> &dict1_;
  const std::vector<std::string> &dict2_;
};

struct DictNotEqual {
  DictNotEqual(const std::vector<std::string> &dict1, const std::vector<std::string> &dict2)
    : dict1_(dict1), dict2_(dict2)
  {
  }

  bool operator()(const uint32_t &a, const uint32_t &b) const
  {
#ifdef DEBUG_PANDAS
    assert(0 <= a && a < dict1_.size());
    assert(0 <= b && b < dict2_.size());
#endif
    return dict1_[a] != dict2_[b];
  }

  const std::vector<std::string> &dict1_;
  const std::vector<std::string> &dict2_;
};

template <BinaryOpCode CODE>
struct DictBinaryOp;

template <>
struct DictBinaryOp<BinaryOpCode::EQUAL> : public DictEqual {
  DictBinaryOp(const std::vector<std::string> &dict1, const std::vector<std::string> &dict2)
    : DictEqual(dict1, dict2)
  {
  }
};

template <>
struct DictBinaryOp<BinaryOpCode::NOT_EQUAL> : public DictNotEqual {
  DictBinaryOp(const std::vector<std::string> &dict1, const std::vector<std::string> &dict2)
    : DictNotEqual(dict1, dict2)
  {
  }
};

struct CategoryBinaryOpImpl {
  template <BinaryOpCode OP_CODE,
            std::enable_if_t<OP_CODE == BinaryOpCode::EQUAL || OP_CODE == BinaryOpCode::NOT_EQUAL>
              * = nullptr>
  ColumnView operator()(const ColumnView &in1, const ColumnView &in2, alloc::Allocator &allocator)
  {
    std::vector<std::string> dict1;
    std::vector<std::string> dict2;

    pandas::category::to_dictionary(dict1, in1.child(1));
    pandas::category::to_dictionary(dict2, in2.child(1));

    if (dict1 == dict2) {
      ColumnView in1_codes(
        in1.child(0).code(), in1.child(0).raw_column(), in1.size(), in1.raw_bitmask());
      ColumnView in2_codes(
        in2.child(0).code(), in2.child(0).raw_column(), in2.size(), in2.raw_bitmask());
      return numeric::binary_op(OP_CODE, in1_codes, in2_codes, allocator);
    }

    using ARG = uint32_t;
    using OP  = DictBinaryOp<OP_CODE>;
    using RES = std::result_of_t<OP(ARG, ARG)>;

    OP op{dict1, dict2};
    auto res_code = pandas_type_code_of<RES>;

    auto size  = in1.size();
    auto out   = allocator.allocate_elements<RES>(size);
    auto p_in1 = in1.child(0).column<ARG>();
    auto p_in2 = in2.child(0).column<ARG>();

    if (!in1.nullable() && !in2.nullable()) {
      for (auto idx = 0; idx < size; ++idx) out[idx] = op(p_in1[idx], p_in2[idx]);
      return ColumnView(res_code, out, size);
    } else if (in1.nullable() || in2.nullable()) {
      Bitmask in_b(size, allocator);
      in_b.set_all_valid();
      if (in1.nullable()) intersect_bitmasks(in_b, in_b, in1.bitmask());
      if (in2.nullable()) intersect_bitmasks(in_b, in_b, in2.bitmask());

      for (auto idx = 0; idx < size; ++idx) {
        bool valid = in_b.get(idx);
        if (valid)
          out[idx] = op(p_in1[idx], p_in2[idx]);
        else
          out[idx] = valid != (OP_CODE == BinaryOpCode::NOT_EQUAL);
      }
      return ColumnView{res_code, out, size};
    }
    assert(false);
    return ColumnView();
  }

  template <BinaryOpCode OP_CODE,
            std::enable_if_t<!(OP_CODE == BinaryOpCode::EQUAL ||
                               OP_CODE == BinaryOpCode::NOT_EQUAL)> * = nullptr>
  ColumnView operator()(const ColumnView &in1, const ColumnView &in2, alloc::Allocator &allocator)
  {
    assert(false);
    return ColumnView{};
  }
};

ColumnView binary_op(BinaryOpCode op_code,
                     const ColumnView &in1,
                     const ColumnView &in2,
                     alloc::Allocator &allocator)
{
  return type_dispatch(op_code, CategoryBinaryOpImpl{}, in1, in2, allocator);
}

}  // namespace category

ColumnView binary_op(BinaryOpCode op_code,
                     const ColumnView &in1,
                     const ColumnView &in2,
                     alloc::Allocator &allocator)
{
  if (is_primitive_type_code(in1.code()))
    return numeric::binary_op(op_code, in1, in2, allocator);
  else if (in1.code() == TypeCode::STRING)
    return string::binary_op(op_code, in1, in2, allocator);
  else if (in1.code() == TypeCode::CAT32)
    return category::binary_op(op_code, in1, in2, allocator);
  else {
    assert(false);
    return ColumnView();
  }
}

}  // namespace detail

/*static*/ void BinaryOpTask::cpu_variant(const Task *task,
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

  alloc::DeferredBufferAllocator allocator;
  auto result = detail::binary_op(op_code, in1.view(), in2.view(), allocator);
  out.return_from_view(allocator, result);
}

static void __attribute__((constructor)) register_tasks(void) { BinaryOpTask::register_variants(); }

}  // namespace binaryop
}  // namespace pandas
}  // namespace legate

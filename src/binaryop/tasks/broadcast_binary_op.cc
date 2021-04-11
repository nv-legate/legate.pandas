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
#include "binaryop/binary_op.h"
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
  ColumnView operator()(const ColumnView &in1,
                        const Scalar &in2,
                        bool scalar_on_rhs,
                        alloc::Allocator &allocator)
  {
    using ARG = pandas_type_of<TYPE_CODE>;
    using OP  = BinaryOp<OP_CODE, ARG>;
    using RES = std::result_of_t<OP(ARG, ARG)>;

    OP op{};
    auto res_code = pandas_type_code_of<RES>;

    auto size = in1.size();
    auto out  = allocator.allocate_elements<RES>(size);

    // Special case when the RHS scalar is invalid
    if (!in2.valid()) {
      Bitmask out_b(size, allocator);
      out_b.clear();
      return ColumnView(res_code, out, size, out_b.raw_ptr());
    }

    auto p_in1  = in1.column<ARG>();
    auto scalar = in2.value<ARG>();

    if (!in1.nullable()) {
      if (scalar_on_rhs)
        for (auto idx = 0; idx < size; ++idx) out[idx] = op(p_in1[idx], scalar);
      else
        for (auto idx = 0; idx < size; ++idx) out[idx] = op(scalar, p_in1[idx]);
      return ColumnView(res_code, out, size);
    } else {
      auto in_b = in1.bitmask();

      Bitmask out_b(size, allocator);
      in_b.copy(out_b);

      if (scalar_on_rhs)
        for (auto idx = 0; idx < size; ++idx) {
          bool valid = in_b.get(idx);
          if (valid) out[idx] = op(p_in1[idx], scalar);
        }
      else
        for (auto idx = 0; idx < size; ++idx) {
          bool valid = in_b.get(idx);
          if (valid) out[idx] = op(scalar, p_in1[idx]);
        }
      return ColumnView(res_code, out, size, out_b.raw_ptr());
    }
  }

  template <TypeCode TYPE_CODE,
            std::enable_if_t<!is_numeric_type<TYPE_CODE>::value ||
                             !is_arithmetic_op<OP_CODE>::value> * = nullptr>
  ColumnView operator()(const ColumnView &in1,
                        const Scalar &in2,
                        bool scalar_on_rhs,
                        alloc::Allocator &allocator)
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
  ColumnView operator()(const ColumnView &in1,
                        const Scalar &in2,
                        bool scalar_on_rhs,
                        alloc::Allocator &allocator)
  {
    using ARG = pandas_type_of<TYPE_CODE>;
    using OP  = BinaryOp<OP_CODE, ARG>;
    using RES = std::result_of_t<OP(ARG, ARG)>;

    OP op{};
    auto res_code = pandas_type_code_of<RES>;

    auto size = in1.size();
    auto out  = allocator.allocate_elements<RES>(size);

    // Special case when the RHS scalar is invalid
    if (!in2.valid()) {
      memset(out, OP_CODE == BinaryOpCode::NOT_EQUAL, size);
      return ColumnView(res_code, out, size);
    }

    auto p_in1  = in1.column<ARG>();
    auto scalar = in2.value<ARG>();

    if (!in1.nullable()) {
      if (scalar_on_rhs)
        for (auto idx = 0; idx < size; ++idx) out[idx] = op(p_in1[idx], scalar);
      else
        for (auto idx = 0; idx < size; ++idx) out[idx] = op(scalar, p_in1[idx]);
      return ColumnView(res_code, out, size);
    } else {
      auto in_b = in1.bitmask();

      if (scalar_on_rhs)
        for (auto idx = 0; idx < size; ++idx) {
          bool valid = in_b.get(idx);
          if (valid)
            out[idx] = op(p_in1[idx], scalar);
          else
            out[idx] = valid != (OP_CODE == BinaryOpCode::NOT_EQUAL);
        }
      else
        for (auto idx = 0; idx < size; ++idx) {
          bool valid = in_b.get(idx);
          if (valid)
            out[idx] = op(scalar, p_in1[idx]);
          else
            out[idx] = valid != (OP_CODE == BinaryOpCode::NOT_EQUAL);
        }
      return ColumnView{res_code, out, size};
    }
  }

  template <TypeCode TYPE_CODE,
            std::enable_if_t<!is_primitive_type<TYPE_CODE>::value ||
                             is_arithmetic_op<OP_CODE>::value> * = nullptr>
  ColumnView operator()(const ColumnView &in1,
                        const Scalar &in2,
                        bool scalar_on_rhs,
                        alloc::Allocator &allocator)
  {
    assert(false);
    return ColumnView(TYPE_CODE);
  }
};

struct NumericBinaryOpImpl {
  template <BinaryOpCode OP>
  ColumnView operator()(const ColumnView &in1,
                        const Scalar &in2,
                        bool scalar_on_rhs,
                        alloc::Allocator &allocator)
  {
    if (is_arithmetic_op<OP>::value)
      return type_dispatch(
        in1.code(), ArithmeticBinaryOpImpl<OP>{}, in1, in2, scalar_on_rhs, allocator);
    else
      return type_dispatch(
        in1.code(), ComparisonBinaryOpImpl<OP>{}, in1, in2, scalar_on_rhs, allocator);
  }
};

ColumnView broadcast_binary_op(BinaryOpCode op_code,
                               const ColumnView &in1,
                               const Scalar &in2,
                               bool scalar_on_rhs,
                               alloc::Allocator &allocator)
{
  return type_dispatch(op_code, NumericBinaryOpImpl{}, in1, in2, scalar_on_rhs, allocator);
}

}  // namespace numeric

namespace string {

struct StringBinaryOpImpl {
  template <BinaryOpCode OP_CODE, std::enable_if_t<!is_arithmetic_op<OP_CODE>::value> * = nullptr>
  ColumnView operator()(const ColumnView &in1,
                        const Scalar &in2,
                        bool scalar_on_rhs,
                        alloc::Allocator &allocator)
  {
    using ARG = std::string;
    using OP  = BinaryOp<OP_CODE, ARG>;
    using RES = std::result_of_t<OP(ARG, ARG)>;

    OP op{};
    auto res_code = pandas_type_code_of<RES>;

    auto size = in1.size();
    auto out  = allocator.allocate_elements<RES>(size);

    auto in1_o = in1.child(0).column<int32_t>();
    auto in1_c = in1.child(1).column<int8_t>();

    auto scalar = in2.value<ARG>();

    // Special case when the RHS scalar is invalid
    if (!in2.valid()) {
      memset(out, OP_CODE == BinaryOpCode::NOT_EQUAL, size);
      return ColumnView(res_code, out, size);
    }

    if (!in1.nullable()) {
      if (scalar_on_rhs)
        for (auto idx = 0; idx < size; ++idx) {
          std::string v(&in1_c[in1_o[idx]], &in1_c[in1_o[idx + 1]]);
          out[idx] = op(v, scalar);
        }
      else
        for (auto idx = 0; idx < size; ++idx) {
          std::string v(&in1_c[in1_o[idx]], &in1_c[in1_o[idx + 1]]);
          out[idx] = op(scalar, v);
        }
      return ColumnView(res_code, out, size);
    } else {
      auto in_b = in1.bitmask();

      if (scalar_on_rhs)
        for (auto idx = 0; idx < size; ++idx) {
          bool valid = in_b.get(idx);
          if (valid) {
            std::string v(&in1_c[in1_o[idx]], &in1_c[in1_o[idx + 1]]);
            out[idx] = op(v, scalar);
          } else
            out[idx] = valid != (OP_CODE == BinaryOpCode::NOT_EQUAL);
        }
      else
        for (auto idx = 0; idx < size; ++idx) {
          bool valid = in_b.get(idx);
          if (valid) {
            std::string v(&in1_c[in1_o[idx]], &in1_c[in1_o[idx + 1]]);
            out[idx] = op(scalar, v);
          } else
            out[idx] = valid != (OP_CODE == BinaryOpCode::NOT_EQUAL);
        }
      return ColumnView{res_code, out, size};
    }
  }

  template <BinaryOpCode OP_CODE, std::enable_if_t<is_arithmetic_op<OP_CODE>::value> * = nullptr>
  ColumnView operator()(const ColumnView &in1,
                        const Scalar &in2,
                        bool scalar_on_rhs,
                        alloc::Allocator &allocator)
  {
    assert(false);
    return ColumnView();
  }
};

ColumnView broadcast_binary_op(BinaryOpCode op_code,
                               const ColumnView &in1,
                               const Scalar &in2,
                               bool scalar_on_rhs,
                               alloc::Allocator &allocator)
{
  return type_dispatch(op_code, StringBinaryOpImpl{}, in1, in2, scalar_on_rhs, allocator);
}

}  // namespace string

namespace category {

ColumnView broadcast_binary_op(BinaryOpCode op_code,
                               const ColumnView &in1,
                               const Scalar &in2,
                               bool scalar_on_rhs,
                               alloc::Allocator &allocator)
{
  ColumnView in1_codes(
    in1.child(0).code(), in1.child(0).raw_column(), in1.size(), in1.raw_bitmask());

  return numeric::broadcast_binary_op(op_code, in1_codes, in2, scalar_on_rhs, allocator);
}

}  // namespace category

ColumnView broadcast_binary_op(BinaryOpCode op_code,
                               const ColumnView &in1,
                               const Scalar &in2,
                               bool scalar_on_rhs,
                               alloc::Allocator &allocator)
{
  if (is_primitive_type_code(in1.code()))
    return numeric::broadcast_binary_op(op_code, in1, in2, scalar_on_rhs, allocator);
  else if (in1.code() == TypeCode::STRING)
    return string::broadcast_binary_op(op_code, in1, in2, scalar_on_rhs, allocator);
  else if (in1.code() == TypeCode::CAT32)
    return category::broadcast_binary_op(op_code, in1, in2, scalar_on_rhs, allocator);
  else {
    assert(false);
    return ColumnView{in1.code()};
  }
}

}  // namespace detail

/*static*/ void BroadcastBinaryOpTask::cpu_variant(const Task *task,
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

  alloc::DeferredBufferAllocator allocator;
  auto result = detail::broadcast_binary_op(op_code, in.view(), scalar, scalar_on_rhs, allocator);
  out.return_from_view(allocator, result);
}

static void __attribute__((constructor)) register_tasks(void)
{
  BroadcastBinaryOpTask::register_variants();
}

}  // namespace binaryop
}  // namespace pandas
}  // namespace legate

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

#include "binaryop/tasks/scalar_binary_op.h"
#include "binaryop/binary_op.h"
#include "util/type_dispatch.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace binaryop {

using namespace Legion;

namespace detail {

namespace numeric {

template <BinaryOpCode OP_CODE>
struct ComparisonBinaryOpImpl {
  template <TypeCode TYPE_CODE,
            std::enable_if_t<is_primitive_type<TYPE_CODE>::value &&
                             !is_arithmetic_op<OP_CODE>::value> * = nullptr>
  bool operator()(const void *in1, const void *in2)
  {
    using ARG = pandas_type_of<TYPE_CODE>;
    using OP  = BinaryOp<OP_CODE, ARG>;
    using RES = std::result_of_t<OP(ARG, ARG)>;

    OP op{};
    return op(*static_cast<const ARG *>(in1), *static_cast<const ARG *>(in2));
  }

  template <TypeCode TYPE_CODE,
            std::enable_if_t<!is_primitive_type<TYPE_CODE>::value ||
                             is_arithmetic_op<OP_CODE>::value> * = nullptr>
  bool operator()(const void *in1, const void *in2)
  {
    assert(false);
    return false;
  }
};

struct NumericBinaryOpImpl {
  template <BinaryOpCode OP>
  bool operator()(TypeCode type_code, const void *in1, const void *in2)
  {
    if (!is_arithmetic_op<OP>::value)
      return type_dispatch(type_code, ComparisonBinaryOpImpl<OP>{}, in1, in2);
    else {
      assert(false);
      return false;
    }
  }
};

bool scalar_binary_op(BinaryOpCode op_code, TypeCode type_code, const void *in1, const void *in2)
{
  return type_dispatch(op_code, NumericBinaryOpImpl{}, type_code, in1, in2);
}

}  // namespace numeric

bool scalar_binary_op(BinaryOpCode op_code, TypeCode type_code, const void *in1, const void *in2)
{
  if (is_primitive_type_code(type_code))
    return numeric::scalar_binary_op(op_code, type_code, in1, in2);
  else {
    assert(false);
    return false;
  }
}

}  // namespace detail

/*static*/ bool ScalarBinaryOpTask::cpu_variant(const Task *task,
                                                const std::vector<PhysicalRegion> &regions,
                                                Context context,
                                                Runtime *runtime)
{
  Deserializer ctx{task, regions};

  BinaryOpCode op_code;
  TypeCode type_code;
  FromRawFuture in1;
  FromRawFuture in2;

  deserialize(ctx, op_code);
  deserialize(ctx, type_code);
  deserialize(ctx, in1);
  deserialize(ctx, in2);

  assert(type_code != TypeCode::STRING);
  assert(op_code == BinaryOpCode::EQUAL);

  return detail::scalar_binary_op(op_code, type_code, in1.rawptr_, in2.rawptr_);
}

static void __attribute__((constructor)) register_tasks(void)
{
  ScalarBinaryOpTask::register_variants_with_return<bool, bool>();
}

}  // namespace binaryop
}  // namespace pandas
}  // namespace legate

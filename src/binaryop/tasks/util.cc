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

#include "binaryop/tasks/util.h"

namespace legate {
namespace pandas {
namespace binaryop {
namespace detail {

cudf::binary_operator to_cudf_binary_operator(BinaryOpCode op_code, TypeCode type_code)
{
  switch (op_code) {
    case BinaryOpCode::ADD: {
      return cudf::binary_operator::ADD;
    }
    case BinaryOpCode::SUB: {
      return cudf::binary_operator::SUB;
    }
    case BinaryOpCode::MUL: {
      return cudf::binary_operator::MUL;
    }
    case BinaryOpCode::DIV: {
      return cudf::binary_operator::DIV;
    }
    case BinaryOpCode::FLOOR_DIV: {
      return cudf::binary_operator::FLOOR_DIV;
    }
    case BinaryOpCode::MOD: {
      if (type_code == TypeCode::FLOAT || type_code == TypeCode::DOUBLE)
        return cudf::binary_operator::PYMOD;
      else
        return cudf::binary_operator::MOD;
    }
    case BinaryOpCode::POW: {
      return cudf::binary_operator::POW;
    }
    case BinaryOpCode::EQUAL: {
      return cudf::binary_operator::EQUAL;
    }
    case BinaryOpCode::NOT_EQUAL: {
      return cudf::binary_operator::NOT_EQUAL;
    }
    case BinaryOpCode::LESS: {
      return cudf::binary_operator::LESS;
    }
    case BinaryOpCode::GREATER: {
      return cudf::binary_operator::GREATER;
    }
    case BinaryOpCode::LESS_EQUAL: {
      return cudf::binary_operator::LESS_EQUAL;
    }
    case BinaryOpCode::GREATER_EQUAL: {
      return cudf::binary_operator::GREATER_EQUAL;
    }
    case BinaryOpCode::BITWISE_AND: {
      return cudf::binary_operator::BITWISE_AND;
    }
    case BinaryOpCode::BITWISE_OR: {
      return cudf::binary_operator::BITWISE_OR;
    }
    case BinaryOpCode::BITWISE_XOR: {
      return cudf::binary_operator::BITWISE_XOR;
    }
    default: {
      assert(false);
      return cudf::binary_operator::INVALID_BINARY;
    }
  }
}

}  // namespace detail
}  // namespace binaryop
}  // namespace pandas
}  // namespace legate

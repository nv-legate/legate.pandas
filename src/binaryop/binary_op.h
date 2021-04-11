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

#pragma once

#include <cmath>

#include "pandas.h"

namespace legate {
namespace pandas {
namespace binaryop {

template <BinaryOpCode CODE>
struct is_arithmetic_op : std::true_type {
};

template <>
struct is_arithmetic_op<BinaryOpCode::EQUAL> : std::false_type {
};

template <>
struct is_arithmetic_op<BinaryOpCode::NOT_EQUAL> : std::false_type {
};

template <>
struct is_arithmetic_op<BinaryOpCode::LESS> : std::false_type {
};

template <>
struct is_arithmetic_op<BinaryOpCode::GREATER> : std::false_type {
};

template <>
struct is_arithmetic_op<BinaryOpCode::LESS_EQUAL> : std::false_type {
};

template <>
struct is_arithmetic_op<BinaryOpCode::GREATER_EQUAL> : std::false_type {
};

template <BinaryOpCode OP_CODE, typename T>
struct BinaryOp;

template <typename T>
struct BinaryOp<BinaryOpCode::ADD, T> : std::plus<T> {
};

template <typename T>
struct BinaryOp<BinaryOpCode::SUB, T> : std::minus<T> {
};

template <typename T>
struct BinaryOp<BinaryOpCode::MUL, T> : std::multiplies<T> {
};

template <typename T>
struct BinaryOp<BinaryOpCode::DIV, T> {
  template <typename _T = T, std::enable_if_t<std::is_integral<_T>::value>* = nullptr>
  constexpr double operator()(const _T& a, const _T& b) const
  {
    return static_cast<double>(a) / static_cast<double>(b);
  }

  template <typename _T = T, std::enable_if_t<!std::is_integral<_T>::value>* = nullptr>
  constexpr _T operator()(const _T& a, const _T& b) const
  {
    return a / b;
  }
};

template <typename T>
struct BinaryOp<BinaryOpCode::FLOOR_DIV, T> {
  constexpr T operator()(const T& a, const T& b) const { return static_cast<T>(std::floor(a / b)); }
};

template <typename T>
struct BinaryOp<BinaryOpCode::MOD, T> {
  template <typename _T = T, std::enable_if_t<std::is_integral<_T>::value>* = nullptr>
  constexpr _T operator()(const _T& a, const _T& b) const
  {
    return a % b;
  }

  template <typename _T = T, std::enable_if_t<!std::is_integral<_T>::value>* = nullptr>
  constexpr _T operator()(const _T& a, const _T& b) const
  {
    _T res = std::fmod(a, b);
    if (res) {
      if ((b < _T{0}) != (res < _T{0})) res += b;
    } else {
      res = std::copysign(_T{0}, b);
    }
    return res;
  }
};

template <typename T>
struct BinaryOp<BinaryOpCode::POW, T> {
  constexpr T operator()(const T& base, const T& exponent) const
  {
    return static_cast<T>(std::pow(base, exponent));
  }
};

template <typename T>
struct BinaryOp<BinaryOpCode::EQUAL, T> : std::equal_to<T> {
};

template <typename T>
struct BinaryOp<BinaryOpCode::NOT_EQUAL, T> : std::not_equal_to<T> {
};

template <typename T>
struct BinaryOp<BinaryOpCode::LESS, T> : std::less<T> {
};

template <typename T>
struct BinaryOp<BinaryOpCode::GREATER, T> : std::greater<T> {
};

template <typename T>
struct BinaryOp<BinaryOpCode::LESS_EQUAL, T> : std::less_equal<T> {
};

template <typename T>
struct BinaryOp<BinaryOpCode::GREATER_EQUAL, T> : std::greater_equal<T> {
};

template <typename T>
struct BinaryOp<BinaryOpCode::BITWISE_AND, T> {
  template <typename _T = T, std::enable_if_t<std::is_integral<_T>::value>* = nullptr>
  constexpr _T operator()(const _T& a, const _T& b) const
  {
    return a & b;
  }

  template <typename _T = T, std::enable_if_t<!std::is_integral<_T>::value>* = nullptr>
  constexpr _T operator()(const _T& a, const _T& b) const
  {
    assert(false);
    return a + b;
  }
};

template <typename T>
struct BinaryOp<BinaryOpCode::BITWISE_OR, T> {
  template <typename _T = T, std::enable_if_t<std::is_integral<_T>::value>* = nullptr>
  constexpr _T operator()(const _T& a, const _T& b) const
  {
    return a | b;
  }

  template <typename _T = T, std::enable_if_t<!std::is_integral<_T>::value>* = nullptr>
  constexpr _T operator()(const _T& a, const _T& b) const
  {
    assert(false);
    return a + b;
  }
};

template <typename T>
struct BinaryOp<BinaryOpCode::BITWISE_XOR, T> {
  template <typename _T = T, std::enable_if_t<std::is_integral<_T>::value>* = nullptr>
  constexpr _T operator()(const _T& a, const _T& b) const
  {
    return a ^ b;
  }

  template <typename _T = T, std::enable_if_t<!std::is_integral<_T>::value>* = nullptr>
  constexpr _T operator()(const _T& a, const _T& b) const
  {
    assert(false);
    return a + b;
  }
};

}  // namespace binaryop
}  // namespace pandas
}  // namespace legate

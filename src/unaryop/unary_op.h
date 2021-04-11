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
namespace unaryop {

template <UnaryOpCode OP_CODE, typename T>
struct UnaryOp;

template <typename T>
struct UnaryOp<UnaryOpCode::ABS, T> {
  template <class _T                                                                     = T,
            std::enable_if_t<std::is_integral<_T>::value and std::is_signed<_T>::value>* = nullptr>
  constexpr _T operator()(const _T& x) const
  {
    return std::abs(x);
  }

  template <
    class _T                                                                       = T,
    std::enable_if_t<std::is_integral<_T>::value and std::is_unsigned<_T>::value>* = nullptr>
  constexpr _T operator()(const _T& x) const
  {
    return x;
  }

  template <class _T = T, std::enable_if_t<!std::is_integral<_T>::value>* = nullptr>
  constexpr _T operator()(const _T& x) const
  {
    return std::fabs(x);
  }
};

template <typename T>
struct UnaryOp<UnaryOpCode::BIT_INVERT, T> {
  template <class _T = T, std::enable_if_t<std::is_integral<_T>::value>* = nullptr>
  constexpr _T operator()(const _T& x) const
  {
    return ~x;
  }

  template <class _T = T, std::enable_if_t<!std::is_integral<_T>::value>* = nullptr>
  constexpr _T operator()(const _T& x) const
  {
    assert(false);
    return x;
  }
};

template <>
struct UnaryOp<UnaryOpCode::BIT_INVERT, bool> {
  constexpr bool operator()(const bool& x) const { return !x; }
};

}  // namespace unaryop
}  // namespace pandas
}  // namespace legate

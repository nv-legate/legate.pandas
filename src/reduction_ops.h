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

#include "pandas.h"

namespace legate {
namespace pandas {

struct RangeUnion {
  using VAL = Legion::Rect<1>;
  using LHS = VAL;
  using RHS = VAL;

  static const RHS identity;
  static const int REDOP_ID = PANDAS_REDOP_RANGE_UNION;

  template <bool EXCLUSIVE>
  static void apply(LHS &lhs, RHS rhs);

  template <bool EXCLUSIVE>
  static void fold(RHS &rhs1, RHS rhs2);
};

}  // namespace pandas
}  // namespace legate

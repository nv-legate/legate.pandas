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

#include <limits>

#include "reduction_ops.h"

namespace legate {
namespace pandas {

using namespace Legion;

/*static*/ const RangeUnion::VAL RangeUnion::identity{
  Point<1>{std::numeric_limits<int64_t>::max()}, Point<1>{std::numeric_limits<int64_t>::min()}};

template <>
/*static*/ void RangeUnion::apply<true>(RangeUnion::LHS &lhs, RangeUnion::RHS rhs)
{
  Legion::MinReduction<int64_t>::apply<true>(reinterpret_cast<int64_t &>(lhs.lo[0]),
                                             reinterpret_cast<int64_t &>(rhs.lo[0]));
  Legion::MaxReduction<int64_t>::apply<true>(reinterpret_cast<int64_t &>(lhs.hi[0]),
                                             reinterpret_cast<int64_t &>(rhs.hi[0]));
}

template <>
/*static*/ void RangeUnion::apply<false>(RangeUnion::LHS &lhs, RangeUnion::RHS rhs)
{
  Legion::MinReduction<int64_t>::apply<false>(reinterpret_cast<int64_t &>(lhs.lo[0]),
                                              reinterpret_cast<int64_t &>(rhs.lo[0]));
  Legion::MaxReduction<int64_t>::apply<false>(reinterpret_cast<int64_t &>(lhs.hi[0]),
                                              reinterpret_cast<int64_t &>(rhs.hi[0]));
}

template <>
/*static*/ void RangeUnion::fold<true>(RangeUnion::LHS &lhs, RangeUnion::RHS rhs)
{
  Legion::MinReduction<int64_t>::fold<true>(reinterpret_cast<int64_t &>(lhs.lo[0]),
                                            reinterpret_cast<int64_t &>(rhs.lo[0]));
  Legion::MaxReduction<int64_t>::fold<true>(reinterpret_cast<int64_t &>(lhs.hi[0]),
                                            reinterpret_cast<int64_t &>(rhs.hi[0]));
}

template <>
/*static*/ void RangeUnion::fold<false>(RangeUnion::LHS &lhs, RangeUnion::RHS rhs)
{
  Legion::MinReduction<int64_t>::fold<false>(reinterpret_cast<int64_t &>(lhs.lo[0]),
                                             reinterpret_cast<int64_t &>(rhs.lo[0]));
  Legion::MaxReduction<int64_t>::fold<false>(reinterpret_cast<int64_t &>(lhs.hi[0]),
                                             reinterpret_cast<int64_t &>(rhs.hi[0]));
}

}  // namespace pandas
}  // namespace legate

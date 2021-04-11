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

#include "copy/copy_if_else.h"
#include "copy/fill.h"
#include "util/type_dispatch.h"

namespace legate {
namespace pandas {
namespace copy {

using namespace Legion;
using ColumnView = detail::Column;

namespace detail {

struct CopyIfElse {
  template <TypeCode CODE, std::enable_if_t<is_primitive_type<CODE>::value> * = nullptr>
  ColumnView operator()(const ColumnView &cond,
                        const ColumnView &in,
                        const ColumnView &other,
                        bool negate,
                        alloc::Allocator &allocator)
  {
    using VAL = pandas_type_of<CODE>;

    auto size = in.size();

    auto p_cond  = cond.column<bool>();
    auto p_in    = in.column<VAL>();
    auto p_other = other.column<VAL>();

    auto cond_b  = cond.nullable() ? cond.bitmask() : Bitmask(size, allocator, true, true);
    auto in_b    = in.nullable() ? in.bitmask() : Bitmask(size, allocator, true, true);
    auto other_b = other.nullable() ? other.bitmask() : Bitmask(size, allocator, true, true);

    auto p_out = allocator.allocate_elements<VAL>(size);
    Bitmask out_b(size, allocator);

    for (auto idx = 0; idx < size; ++idx) {
      auto valid = cond_b.get(idx) && p_cond[idx];
      if (!negate == valid) {
        p_out[idx] = p_in[idx];
        out_b.set(idx, in_b.get(idx));
      } else {
        p_out[idx] = p_other[idx];
        out_b.set(idx, other_b.get(idx));
      }
    }
    return ColumnView(CODE, p_out, size, out_b.raw_ptr());
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::STRING> * = nullptr>
  ColumnView operator()(const ColumnView &cond,
                        const ColumnView &in,
                        const ColumnView &other,
                        bool negate,
                        alloc::Allocator &allocator)
  {
    auto size = in.size();

    auto p_cond = cond.column<bool>();

    auto cond_b  = cond.nullable() ? cond.bitmask() : Bitmask(size, allocator, true, true);
    auto in_b    = in.nullable() ? in.bitmask() : Bitmask(size, allocator, true, true);
    auto other_b = other.nullable() ? other.bitmask() : Bitmask(size, allocator, true, true);

    auto p_in_o    = in.child(0).column<int32_t>();
    auto p_other_o = other.child(0).column<int32_t>();

    auto p_in_c    = in.child(1).column<int8_t>();
    auto p_other_c = other.child(1).column<int8_t>();

    size_t num_chars = 0;
    for (auto idx = 0; idx < size; ++idx) {
      auto valid = cond_b.get(idx) && p_cond[idx];
      num_chars +=
        (!negate == valid) ? p_in_o[idx + 1] - p_in_o[idx] : p_other_o[idx + 1] - p_other_o[idx];
    }

    auto p_out_o = allocator.allocate_elements<int32_t>(size + 1);
    auto p_out_c = allocator.allocate_elements<int8_t>(num_chars);

    Bitmask out_b(size, allocator);

    int32_t curr_off = 0;
    for (auto idx = 0; idx < size; ++idx) {
      auto valid   = cond_b.get(idx) && p_cond[idx];
      p_out_o[idx] = curr_off;
      if (!negate == valid) {
        int32_t len = p_in_o[idx + 1] - p_in_o[idx];
        memcpy(&p_out_c[curr_off], &p_in_c[p_in_o[idx]], len);
        curr_off += len;
        out_b.set(idx, in_b.get(idx));
      } else {
        int32_t len = p_other_o[idx + 1] - p_other_o[idx];
        memcpy(&p_out_c[curr_off], &p_other_c[p_other_o[idx]], len);
        curr_off += len;
        out_b.set(idx, other_b.get(idx));
      }
    }
    p_out_o[size] = curr_off;

    return ColumnView(TypeCode::STRING,
                      nullptr,
                      size,
                      out_b.raw_ptr(),
                      {ColumnView(TypeCode::INT32, p_out_o, size + 1),
                       ColumnView(TypeCode::INT8, p_out_c, num_chars)});
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::CAT32> * = nullptr>
  ColumnView operator()(const ColumnView &cond,
                        const ColumnView &in,
                        const ColumnView &other,
                        bool negate,
                        alloc::Allocator &allocator)
  {
    assert(false);
  }
};

}  // namespace detail

ColumnView copy_if_else(const ColumnView &cond,
                        const ColumnView &in,
                        const ColumnView &other,
                        bool negate,
                        alloc::Allocator &allocator)
{
#ifdef DEBUG_PANDAS
  assert(in.code() == other.code());
  assert(in.size() == other.size());
#endif

  return type_dispatch(in.code(), detail::CopyIfElse{}, cond, in, other, negate, allocator);
}

ColumnView copy_if_else(const ColumnView &cond,
                        const ColumnView &in,
                        const Scalar &other,
                        bool negate,
                        alloc::Allocator &allocator)
{
  auto other_repeat = fill(other, in.size(), allocator);
  return copy_if_else(cond, in, other_repeat, negate, allocator);
}

ColumnView copy_if_else(const ColumnView &cond,
                        const ColumnView &in,
                        bool negate,
                        alloc::Allocator &allocator)
{
  auto other_repeat = fill(Scalar(in.code()), in.size(), allocator);
  return copy_if_else(cond, in, other_repeat, negate, allocator);
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

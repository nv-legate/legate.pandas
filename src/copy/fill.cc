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

#include "copy/fill.h"
#include "util/type_dispatch.h"

namespace legate {
namespace pandas {
namespace copy {

using ColumnView = detail::Column;

namespace detail {

struct Fill {
  template <TypeCode CODE, std::enable_if_t<is_primitive_type<CODE>::value> * = nullptr>
  ColumnView operator()(const Scalar &scalar, size_t size, alloc::Allocator &allocator)
  {
    using VAL = pandas_type_of<CODE>;

    auto p_out   = allocator.allocate_elements<VAL>(size);
    auto p_out_b = static_cast<Bitmask::AllocType *>(nullptr);

    for (auto idx = 0; idx < size; ++idx) p_out[idx] = scalar.value<VAL>();

    if (!scalar.valid()) {
      p_out_b = allocator.allocate_elements<Bitmask::AllocType>(size);
      Bitmask(p_out_b, size).clear();
    }

    return ColumnView(CODE, p_out, size, p_out_b);
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::STRING> * = nullptr>
  ColumnView operator()(const Scalar &scalar, size_t size, alloc::Allocator &allocator)
  {
    auto p_out_o = static_cast<int32_t *>(nullptr);
    auto p_out_c = static_cast<int8_t *>(nullptr);
    auto p_out_b = static_cast<Bitmask::AllocType *>(nullptr);

    auto num_offsets = size + 1;
    auto num_chars   = 0;

    if (scalar.valid()) {
      const auto value = scalar.value<std::string>();
      const auto len   = value.size();

      num_chars = size * len;
      p_out_o   = allocator.allocate_elements<int32_t>(num_offsets);
      p_out_c   = allocator.allocate_elements<int8_t>(num_chars);

      int32_t curr_off = 0;
      for (auto idx = 0; idx < size; ++idx) {
        p_out_o[idx] = curr_off;
        memcpy(&p_out_c[curr_off], value.c_str(), len);
        curr_off += len;
      }
      p_out_o[size] = curr_off;
    } else {
      p_out_o = allocator.allocate_elements<int32_t>(num_offsets);
      p_out_c = allocator.allocate_elements<int8_t>(0);

      for (auto idx = 0; idx <= size; ++idx) p_out_o[idx] = 0;
      p_out_b = allocator.allocate_elements<Bitmask::AllocType>(size);
      Bitmask(p_out_b, size).clear();
    }

    return ColumnView(TypeCode::STRING,
                      nullptr,
                      size,
                      p_out_b,
                      {
                        ColumnView(TypeCode::INT32, p_out_o, num_offsets),
                        ColumnView(TypeCode::INT8, p_out_c, num_chars),
                      });
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::CAT32> * = nullptr>
  ColumnView operator()(const Scalar &scalar, size_t size, alloc::Allocator &allocator)
  {
    assert(false);
    return ColumnView();
  }
};

}  // namespace detail

ColumnView fill(const Scalar &scalar, size_t size, alloc::Allocator &allocator)
{
  return type_dispatch(scalar.code(), detail::Fill{}, scalar, size, allocator);
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

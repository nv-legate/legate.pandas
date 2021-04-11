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

#include "copy/concatenate.h"
#include "util/type_dispatch.h"

namespace legate {
namespace pandas {

using ColumnView = detail::Column;
using TableView  = detail::Table;

namespace copy {

namespace detail {

Bitmask::AllocType *concatenate_bitmasks(std::vector<ColumnView> inputs,
                                         size_t size,
                                         alloc::Allocator &allocator)
{
  auto out_b   = allocator.allocate_elements<Bitmask::AllocType>(size);
  auto p_out_b = out_b;

  for (auto &&input : inputs) {
    memcpy(p_out_b, input.raw_bitmask(), sizeof(Bitmask::AllocType) * input.size());
    p_out_b += input.size();
  }

  return out_b;
}

struct ConcatenateOp {
  template <TypeCode CODE,
            std::enable_if_t<CODE != TypeCode::STRING && CODE != TypeCode::CAT32> * = nullptr>
  decltype(auto) operator()(const std::vector<ColumnView> inputs, alloc::Allocator &allocator)
  {
    using VAL = pandas_type_of<CODE>;

    size_t out_size = 0;
    bool nullable   = false;

    for (auto &&input : inputs) {
      out_size += input.size();
      nullable = nullable || input.nullable();
    }

    auto out   = allocator.allocate_elements<VAL>(out_size);
    auto p_out = out;

    for (auto &&input : inputs) {
      memcpy(p_out, input.raw_column(), sizeof(VAL) * input.size());
      p_out += input.size();
    }

    Bitmask::AllocType *out_b{nullptr};
    if (nullable) out_b = concatenate_bitmasks(inputs, out_size, allocator);

    return ColumnView(CODE, out, out_size, out_b);
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::STRING> * = nullptr>
  decltype(auto) operator()(const std::vector<ColumnView> inputs, alloc::Allocator &allocator)
  {
    size_t out_size      = 0;
    size_t out_char_size = 0;
    bool nullable        = false;

    for (auto &&input : inputs) {
      auto in_size = input.size();
      auto in_o    = input.child(0).column<int32_t>();

      for (auto idx = 0; idx < in_size; ++idx) out_char_size += in_o[idx + 1] - in_o[idx];

      out_size += in_size;
      nullable = nullable || input.nullable();
    }

    if (out_size == 0)
      return ColumnView(
        CODE, nullptr, 0, nullptr, {ColumnView(TypeCode::INT32), ColumnView(TypeCode::INT8)});

    auto out_o       = allocator.allocate_elements<int32_t>(out_size + 1);
    auto p_out_o     = out_o;
    int32_t curr_off = 0;
    for (auto &&input : inputs) {
      auto in_size = input.size();
      auto in_o    = input.child(0).column<int32_t>();
      for (auto idx = 0; idx < in_size; ++idx, ++p_out_o) {
        auto len = in_o[idx + 1] - in_o[idx];
        *p_out_o = curr_off;
        curr_off += len;
      }
    }
    *p_out_o = curr_off;
    ColumnView offset_column(TypeCode::INT32, out_o, out_size + 1);

    std::vector<ColumnView> chars;
    for (auto &&input : inputs) chars.push_back(input.child(1));
    auto char_column = operator()<TypeCode::INT8>(chars, allocator);

    Bitmask::AllocType *out_b{nullptr};
    if (nullable) out_b = concatenate_bitmasks(inputs, out_size, allocator);

    return ColumnView(CODE, nullptr, out_size, out_b, {offset_column, char_column});
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::CAT32> * = nullptr>
  decltype(auto) operator()(const std::vector<ColumnView> inputs, alloc::Allocator &allocator)
  {
    std::vector<ColumnView> codes;

    for (auto &&input : inputs)
      codes.push_back(ColumnView(
        TypeCode::UINT32, input.child(0).column<uint32_t>(), input.size(), input.raw_bitmask()));

    auto &&concatenated = operator()<TypeCode::UINT32>(codes, allocator);

    return ColumnView(
      CODE,
      nullptr,
      concatenated.size(),
      concatenated.raw_bitmask(),
      {ColumnView(TypeCode::UINT32, concatenated.raw_column(), concatenated.size())});
  }
};

}  // namespace detail

ColumnView concatenate(const std::vector<ColumnView> inputs, alloc::Allocator &allocator)
{
  if (inputs.size() == 1) return inputs.front();

  detail::ConcatenateOp concat;
  return type_dispatch(inputs[0].code(), concat, inputs, allocator);
}

TableView concatenate(const std::vector<TableView> &inputs, alloc::Allocator &allocator)
{
  if (inputs.size() == 1) return inputs.front();

  detail::ConcatenateOp concat;

  auto num_columns = inputs.front().num_columns();
  std::vector<ColumnView> concatenated;
  for (auto col_idx = 0; col_idx < num_columns; ++col_idx) {
    std::vector<ColumnView> columns;
    for (auto &&table : inputs) columns.push_back(table.column(col_idx));
    concatenated.push_back(concatenate(columns, allocator));
  }

  return TableView{std::move(concatenated)};
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

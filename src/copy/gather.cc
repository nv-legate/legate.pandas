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

#include "copy/gather.h"
#include "util/type_dispatch.h"

namespace legate {
namespace pandas {

using namespace Legion;

using ColumnView = detail::Column;

namespace copy {

namespace detail {

template <typename VAL>
void gather(VAL* out, const VAL* in, const std::vector<int64_t>& mapping, bool has_out_of_range)
{
  if (!has_out_of_range)
    for (size_t i = 0; i < mapping.size(); ++i) out[i] = in[mapping[i]];
  else
    for (size_t i = 0; i < mapping.size(); ++i) {
      auto j = mapping[i];
      if (j != -1) out[i] = in[j];
    }
}

void gather_bitmask(const Bitmask& out,
                    const Bitmask& in,
                    const std::vector<int64_t>& mapping,
                    bool has_out_of_range,
                    OutOfRangePolicy out_of_range_policy)
{
  if (!has_out_of_range)
    for (size_t i = 0; i < mapping.size(); ++i) out.set(i, in.get(mapping[i]));
  else if (OutOfRangePolicy::NULLIFY == out_of_range_policy)
    for (size_t i = 0; i < mapping.size(); ++i) {
      auto j = mapping[i];
      if (j != -1)
        out.set(i, in.get(j));
      else
        out.set(i, false);
    }
  else
    for (size_t i = 0; i < mapping.size(); ++i) {
      auto j = mapping[i];
      if (j != -1) out.set(i, in.get(j));
    }
}

void set_bitmask(const Bitmask& out,
                 const std::vector<int64_t>& mapping,
                 OutOfRangePolicy out_of_range_policy)
{
  if (OutOfRangePolicy::NULLIFY == out_of_range_policy)
    for (size_t i = 0; i < mapping.size(); ++i) out.set(i, mapping[i] != -1);
  else
    for (size_t i = 0; i < mapping.size(); ++i) {
      auto j = mapping[i];
      if (j != -1) out.set(i, true);
    }
}

struct GatherCopier {
  template <TypeCode CODE,
            std::enable_if_t<CODE != TypeCode::STRING && CODE != TypeCode::CAT32>* = nullptr>
  ColumnView operator()(const ColumnView& input,
                        const std::vector<int64_t>& mapping,
                        bool has_out_of_range,
                        OutOfRangePolicy out_of_range_policy,
                        alloc::Allocator& allocator)
  {
    using VAL = pandas_type_of<CODE>;

    auto out_size = mapping.size();
    auto out      = allocator.allocate_elements<VAL>(out_size);
    auto in       = input.column<VAL>();

    gather<VAL>(out, in, mapping, has_out_of_range);

    Bitmask::AllocType* raw_bitmask{nullptr};
    if (input.nullable() || has_out_of_range) {
      Bitmask out_bitmask(out_size, allocator);
      if (input.nullable())
        gather_bitmask(
          out_bitmask, input.bitmask(), mapping, has_out_of_range, out_of_range_policy);
      else
        set_bitmask(out_bitmask, mapping, out_of_range_policy);
      raw_bitmask = out_bitmask.raw_ptr();
    }

    return ColumnView(CODE, out, out_size, raw_bitmask);
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::STRING>* = nullptr>
  ColumnView operator()(const ColumnView& input,
                        const std::vector<int64_t>& mapping,
                        bool has_out_of_range,
                        OutOfRangePolicy out_of_range_policy,
                        alloc::Allocator& allocator)
  {
    auto out_size = mapping.size();

    if (out_size == 0)
      return ColumnView(
        CODE, nullptr, 0, nullptr, {ColumnView(TypeCode::INT32), ColumnView(TypeCode::INT8)});

    auto out = allocator.allocate_elements<int32_t>(out_size + 1);
    auto in  = input.child(0).column<int32_t>();

    size_t num_chars = 0;

    if (!has_out_of_range) {
      for (size_t i = 0; i < out_size; ++i) {
        auto j = mapping[i];
        num_chars += in[j + 1] - in[j];
      }
    } else {
      for (size_t i = 0; i < out_size; ++i) {
        auto j = mapping[i];
        if (j != -1) num_chars += in[j + 1] - in[j];
      }
    }

    std::vector<int64_t> char_mapping(num_chars, 0);

    size_t curr_off = 0;
    if (!has_out_of_range) {
      for (size_t i = 0; i < out_size; ++i) {
        out[i] = curr_off;
        auto j = mapping[i];
        for (int32_t char_idx = in [j]; char_idx < in[j + 1]; ++char_idx, ++curr_off)
          char_mapping[curr_off] = char_idx;
      }
    } else {
      for (size_t i = 0; i < out_size; ++i) {
        out[i] = curr_off;
        auto j = mapping[i];
        if (-1 == j) continue;
        for (int32_t char_idx = in [j]; char_idx < in[j + 1]; ++char_idx, ++curr_off)
          char_mapping[curr_off] = char_idx;
      }
    }
    out[out_size] = curr_off;

    auto offset_column = ColumnView(TypeCode::INT32, out, out_size + 1);
    auto char_column   = operator()<TypeCode::INT8>(
      input.child(1), char_mapping, false, OutOfRangePolicy::IGNORE, allocator);

    Bitmask::AllocType* raw_bitmask{nullptr};
    if (input.nullable() || has_out_of_range) {
      Bitmask out_bitmask(out_size, allocator);
      if (input.nullable())
        gather_bitmask(
          out_bitmask, input.bitmask(), mapping, has_out_of_range, out_of_range_policy);
      else
        set_bitmask(out_bitmask, mapping, out_of_range_policy);
      raw_bitmask = out_bitmask.raw_ptr();
    }

    return ColumnView(CODE, nullptr, out_size, raw_bitmask, {offset_column, char_column});
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::CAT32>* = nullptr>
  ColumnView operator()(const ColumnView& input,
                        const std::vector<int64_t>& mapping,
                        bool has_out_of_range,
                        OutOfRangePolicy out_of_range_policy,
                        alloc::Allocator& allocator)
  {
    auto out_size = mapping.size();
    auto out      = allocator.allocate_elements<uint32_t>(out_size);
    auto in       = input.child(0).column<uint32_t>();

    gather<uint32_t>(out, in, mapping, has_out_of_range);

    Bitmask::AllocType* raw_bitmask{nullptr};
    if (input.nullable() || has_out_of_range) {
      Bitmask out_bitmask(out_size, allocator);
      if (input.nullable())
        gather_bitmask(
          out_bitmask, input.bitmask(), mapping, has_out_of_range, out_of_range_policy);
      else
        set_bitmask(out_bitmask, mapping, out_of_range_policy);
      raw_bitmask = out_bitmask.raw_ptr();
    }

    return ColumnView(
      CODE, nullptr, out_size, raw_bitmask, {ColumnView(TypeCode::UINT32, out, out_size)});
  }
};

}  // namespace detail

ColumnView gather(const ColumnView& input,
                  const std::vector<int64_t>& mapping,
                  bool has_out_of_range,
                  OutOfRangePolicy out_of_range_policy,
                  alloc::Allocator& allocator)
{
  detail::GatherCopier copier;
  return type_dispatch(
    input.code(), copier, input, mapping, has_out_of_range, out_of_range_policy, allocator);
}

ColumnView gather(const Rect<1>& rect,
                  const std::vector<int64_t>& mapping,
                  bool has_out_of_range,
                  OutOfRangePolicy out_of_range_policy,
                  alloc::Allocator& allocator)
{
  auto out_size = mapping.size();
  auto out      = allocator.allocate_elements<int64_t>(out_size);
  auto lo       = rect.lo[0];

  for (size_t i = 0; i < out_size; ++i) out[i] = lo + mapping[i];

  return ColumnView(TypeCode::INT64, out, out_size);
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

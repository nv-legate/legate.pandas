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

#include "transform/tasks/astype.h"
#include "category/utilities.h"
#include "datetime/utilities.h"
#include "column/column.h"
#include "string/converter.h"
#include "util/allocator.h"
#include "util/type_dispatch.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace transform {

using namespace Legion;
using ColumnView = detail::Column;

namespace detail {

template <typename RES>
struct FromStringConveter {
  template <
    typename _RES                                                                    = RES,
    std::enable_if_t<std::is_integral<_RES>::value && std::is_signed<_RES>::value> * = nullptr>
  RES operator()(const std::string &input)
  {
    return static_cast<RES>(std::stol(input));
  }

  template <
    typename _RES                                                                     = RES,
    std::enable_if_t<std::is_integral<_RES>::value && !std::is_signed<_RES>::value> * = nullptr>
  RES operator()(const std::string &input)
  {
    return static_cast<RES>(std::stoul(input));
  }

  template <typename _RES = RES, std::enable_if_t<!std::is_integral<_RES>::value> * = nullptr>
  RES operator()(const std::string &input)
  {
    return static_cast<RES>(std::stod(input));
  }
};

template <>
struct FromStringConveter<bool> {
  bool operator()(const std::string &input) { return input == "True"; }
};

template <typename RES>
ColumnView from_string(const ColumnView &in, alloc::Allocator &allocator)
{
  auto size  = in.size();
  auto p_out = allocator.allocate_elements<RES>(size);
  FromStringConveter<RES> converter{};

  if (in.nullable()) {
    auto in_b = in.bitmask();
    for (auto idx = 0; idx < size; ++idx) {
      if (!in_b.get(idx)) continue;
      p_out[idx] = converter(in.element<std::string>(idx));
    }
  } else {
    for (auto idx = 0; idx < size; ++idx) p_out[idx] = converter(in.element<std::string>(idx));
  }

  Bitmask::AllocType *p_out_b = nullptr;
  if (in.nullable()) {
    Bitmask out_b(size, allocator);
    in.bitmask().copy(out_b);
    p_out_b = out_b.raw_ptr();
  }

  return ColumnView(pandas_type_code_of<RES>, p_out, size, p_out_b);
}

struct FromString {
  template <TypeCode TO_TYPE, std::enable_if_t<is_numeric_type<TO_TYPE>::value> * = nullptr>
  ColumnView operator()(const ColumnView &in, alloc::Allocator &allocator)
  {
    return from_string<pandas_type_of<TO_TYPE>>(in, allocator);
  }

  template <TypeCode TO_TYPE, std::enable_if_t<TO_TYPE == TypeCode::TS_NS> * = nullptr>
  ColumnView operator()(const ColumnView &in, alloc::Allocator &allocator)
  {
    assert(false);
    return ColumnView();
  }

  template <TypeCode TO_TYPE, std::enable_if_t<TO_TYPE == TypeCode::STRING> * = nullptr>
  ColumnView operator()(const ColumnView &in, alloc::Allocator &allocator)
  {
    assert(false);
    return ColumnView();
  }

  template <TypeCode TO_TYPE, std::enable_if_t<TO_TYPE == TypeCode::CAT32> * = nullptr>
  ColumnView operator()(const ColumnView &in, alloc::Allocator &allocator)
  {
    assert(false);
    return ColumnView();
  }
};

template <TypeCode TO_TYPE>
struct Cast {
  template <TypeCode FROM_TYPE, std::enable_if_t<is_numeric_type<FROM_TYPE>::value> * = nullptr>
  ColumnView operator()(const ColumnView &in, alloc::Allocator &allocator)
  {
    using ARG = pandas_type_of<FROM_TYPE>;
    using RES = pandas_type_of<TO_TYPE>;

    auto res_code = pandas_type_code_of<RES>;

    auto size = in.size();
    auto out  = allocator.allocate_elements<RES>(size);
    auto p_in = in.column<ARG>();

    for (auto idx = 0; idx < size; ++idx) out[idx] = static_cast<RES>(p_in[idx]);
    return ColumnView(res_code, out, size);
  }

  template <TypeCode FROM_TYPE, std::enable_if_t<FROM_TYPE == TypeCode::TS_NS> * = nullptr>
  ColumnView operator()(const ColumnView &in, alloc::Allocator &allocator)
  {
    // Unreachable
    assert(false);
    return ColumnView();
  }

  template <TypeCode FROM_TYPE, std::enable_if_t<FROM_TYPE == TypeCode::STRING> * = nullptr>
  ColumnView operator()(const ColumnView &in, alloc::Allocator &allocator)
  {
    return type_dispatch(TO_TYPE, FromString{}, in, allocator);
  }

  template <TypeCode FROM_TYPE, std::enable_if_t<FROM_TYPE == TypeCode::CAT32> * = nullptr>
  ColumnView operator()(const ColumnView &in, alloc::Allocator &allocator)
  {
    // Unreachable
    assert(false);
    return ColumnView();
  }
};

template <typename Converter, typename ARG>
ColumnView to_string(const Converter &converter, const ColumnView &in, alloc::Allocator &allocator)
{
  auto size = in.size();
  auto p_in = in.column<ARG>();

  size_t num_chars = 0;
  if (in.nullable()) {
    auto in_b = in.bitmask();
    for (auto idx = 0; idx < size; ++idx) {
      if (!in_b.get(idx)) continue;
      auto s = converter(p_in[idx]);
      num_chars += s.size();
    }
  } else {
    for (auto idx = 0; idx < size; ++idx) {
      auto s = converter(p_in[idx]);
      num_chars += s.size();
    }
  }

  auto p_out_o = allocator.allocate_elements<int32_t>(size + 1);
  auto p_out_c = allocator.allocate_elements<int8_t>(num_chars);

  int32_t curr_off = 0;
  if (in.nullable()) {
    auto in_b = in.bitmask();
    for (auto idx = 0; idx < size; ++idx) {
      p_out_o[idx] = curr_off;
      if (!in_b.get(idx)) continue;
      auto out      = converter(p_in[idx]);
      auto out_size = out.size();
      memcpy(&p_out_c[curr_off], out.c_str(), out_size);
      curr_off += out_size;
    }
    p_out_o[size] = curr_off;
  } else {
    for (auto idx = 0; idx < size; ++idx) {
      p_out_o[idx]  = curr_off;
      auto out      = converter(p_in[idx]);
      auto out_size = out.size();
      memcpy(&p_out_c[curr_off], out.c_str(), out_size);
      curr_off += out_size;
    }
    p_out_o[size] = curr_off;
  }

  Bitmask::AllocType *p_out_b = nullptr;
  if (in.nullable()) {
    Bitmask out_b(size, allocator);
    in.bitmask().copy(out_b);
    p_out_b = out_b.raw_ptr();
  }

  return ColumnView(TypeCode::STRING,
                    nullptr,
                    size,
                    p_out_b,
                    {ColumnView(TypeCode::INT32, p_out_o, size + 1),
                     ColumnView(TypeCode::INT8, p_out_c, num_chars)});
}

struct DictionaryConverter {
  DictionaryConverter(const ColumnView &dict_column)
  {
    category::to_dictionary(dictionary, dict_column);
  }

  const std::string &operator()(uint32_t value) const
  {
#ifdef DEBUG_PANDAS
    assert(0 <= value && value < dictionary.size());
#endif
    return dictionary[value];
  }

  std::vector<std::string> dictionary;
};

struct DateTimeConverter {
  std::string operator()(const int64_t &value) const { return datetime::detail::to_string(value); }
};

struct ToString {
  template <TypeCode FROM_TYPE, std::enable_if_t<is_numeric_type<FROM_TYPE>::value> * = nullptr>
  ColumnView operator()(const ColumnView &in, alloc::Allocator &allocator)
  {
    using ARG       = pandas_type_of<FROM_TYPE>;
    using Converter = string::detail::Converter<ARG>;

    Converter converter{};
    return to_string<Converter, ARG>(converter, in, allocator);
  }

  template <TypeCode FROM_TYPE, std::enable_if_t<FROM_TYPE == TypeCode::CAT32> * = nullptr>
  ColumnView operator()(const ColumnView &in, alloc::Allocator &allocator)
  {
    DictionaryConverter converter(in.child(1));
    ColumnView codes(in.child(0).code(), in.child(0).raw_column(), in.size(), in.raw_bitmask());
    return to_string<DictionaryConverter, uint32_t>(converter, codes, allocator);
  }

  template <TypeCode FROM_TYPE, std::enable_if_t<FROM_TYPE == TypeCode::TS_NS> * = nullptr>
  ColumnView operator()(const ColumnView &in, alloc::Allocator &allocator)
  {
    DateTimeConverter converter;
    return to_string<DateTimeConverter, int64_t>(converter, in, allocator);
  }

  template <TypeCode FROM_TYPE, std::enable_if_t<FROM_TYPE == TypeCode::STRING> * = nullptr>
  ColumnView operator()(const ColumnView &in, alloc::Allocator &allocator)
  {
    assert(false);
    return ColumnView();
  }
};

struct DispatchByTgtType {
  template <TypeCode TO_TYPE, std::enable_if_t<is_numeric_type<TO_TYPE>::value> * = nullptr>
  ColumnView operator()(const ColumnView &in, alloc::Allocator &allocator)
  {
    return type_dispatch(in.code(), Cast<TO_TYPE>{}, in, allocator);
  }

  template <TypeCode TO_TYPE, std::enable_if_t<TO_TYPE == TypeCode::TS_NS> * = nullptr>
  ColumnView operator()(const ColumnView &in, alloc::Allocator &allocator)
  {
    assert(false);
    return ColumnView();
  }

  template <TypeCode TO_TYPE, std::enable_if_t<TO_TYPE == TypeCode::STRING> * = nullptr>
  ColumnView operator()(const ColumnView &in, alloc::Allocator &allocator)
  {
    return type_dispatch(in.code(), ToString{}, in, allocator);
  }

  template <TypeCode TO_TYPE, std::enable_if_t<TO_TYPE == TypeCode::CAT32> * = nullptr>
  ColumnView operator()(const ColumnView &in, alloc::Allocator &allocator)
  {
    assert(false);
    return ColumnView();
  }
};

ColumnView astype(const ColumnView &in, TypeCode to_type, alloc::Allocator &allocator)
{
  return type_dispatch(to_type, DispatchByTgtType{}, in, allocator);
}

}  // namespace detail

/*static*/ void AstypeTask::cpu_variant(const Task *task,
                                        const std::vector<PhysicalRegion> &regions,
                                        Context context,
                                        Runtime *runtime)
{
  Deserializer ctx{task, regions};

  OutputColumn out;
  Column<true> in;

  deserialize(ctx, out);
  deserialize(ctx, in);

  if (in.empty()) {
    out.make_empty(true);
    return;
  }

  alloc::DeferredBufferAllocator allocator;
  auto result = detail::astype(in.view(), out.code(), allocator);
  out.return_from_view(allocator, result);
}

static void __attribute__((constructor)) register_tasks(void) { AstypeTask::register_variants(); }

}  // namespace transform
}  // namespace pandas
}  // namespace legate

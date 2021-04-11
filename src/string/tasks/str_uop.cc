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

#include <cctype>

#include "string/tasks/str_uop.h"
#include "column/column.h"
#include "column/detail/column.h"
#include "util/allocator.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace string {

using namespace Legion;
using ColumnView = detail::Column;

namespace detail {

struct Lower {
  void operator()(int8_t *out, const int8_t *in, int32_t lo, int32_t hi) const
  {
    for (int32_t i = lo; i < hi; ++i) out[i] = std::tolower(in[i]);
  }
};

struct Upper {
  void operator()(int8_t *out, const int8_t *in, int32_t lo, int32_t hi) const
  {
    for (int32_t i = lo; i < hi; ++i) out[i] = std::toupper(in[i]);
  }
};

struct Swapcase {
  void operator()(int8_t *out, const int8_t *in, int32_t lo, int32_t hi) const
  {
    for (int32_t i = lo; i < hi; ++i) {
      auto c = in[i];
      out[i] = std::islower(c) ? std::toupper(c) : std::tolower(c);
    }
  }
};

template <typename OP>
ColumnView unary_op(OP op, const ColumnView &in, alloc::Allocator &allocator)
{
  const auto size = in.size();

  auto in_o = in.child(0).template column<int32_t>();
  auto in_c = in.child(1).template column<int8_t>();

  const auto num_offsets = in.child(0).size();
  const auto num_chars   = in.child(1).size();

  auto out_o = allocator.allocate_elements<int32_t>(num_offsets);
  auto out_c = allocator.allocate_elements<int8_t>(num_chars);

  for (size_t i = 0; i < size; ++i) {
    auto lo = in_o[i];
    auto hi = in_o[i + 1];
    if (lo < hi) op(out_c, in_c, lo, hi);
    out_o[i] = in_o[i];
  }
  out_o[size] = in_o[size];

  return ColumnView(TypeCode::STRING,
                    nullptr,
                    0,
                    nullptr,
                    {ColumnView(TypeCode::INT32, out_o, num_offsets),
                     ColumnView(TypeCode::INT8, out_c, num_chars)});
}

}  // namespace detail

/*static*/ void StrUopTask::cpu_variant(const Task *task,
                                        const std::vector<PhysicalRegion> &regions,
                                        Context context,
                                        Runtime *runtime)
{
  Deserializer ctx{task, regions};

  int32_t code;
  deserialize(ctx, code);
  const StringMethods op = static_cast<StringMethods>(code);

  OutputColumn out;
  Column<true> in;
  deserialize(ctx, out);
  deserialize(ctx, in);

  if (in.empty()) {
    out.make_empty(true);
    return;
  }

  alloc::DeferredBufferAllocator allocator;

  switch (op) {
    case StringMethods::LOWER: {
      out.return_from_view(allocator, detail::unary_op(detail::Lower{}, in.view(), allocator));
      break;
    }
    case StringMethods::UPPER: {
      out.return_from_view(allocator, detail::unary_op(detail::Upper{}, in.view(), allocator));
      break;
    }
    case StringMethods::SWAPCASE: {
      out.return_from_view(allocator, detail::unary_op(detail::Swapcase{}, in.view(), allocator));
      break;
    }
    default: {
      assert(false);
      break;
    }
  }
}

static void __attribute__((constructor)) register_tasks(void) { StrUopTask::register_variants(); }

}  // namespace string
}  // namespace pandas
}  // namespace legate

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

#include "transform/tasks/broadcast_fillna.h"
#include "column/column.h"
#include "util/allocator.h"
#include "util/type_dispatch.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace transform {

using namespace Legion;
using ColumnView = detail::Column;

namespace detail {

struct BroadcastFillNa {
  template <TypeCode TYPE_CODE, std::enable_if_t<is_primitive_type<TYPE_CODE>::value> * = nullptr>
  ColumnView operator()(const ColumnView &in, const Scalar &scalar, alloc::Allocator &allocator)
  {
    using ARG = pandas_type_of<TYPE_CODE>;
    using RES = ARG;

    auto size = in.size();
    auto out  = allocator.allocate_elements<RES>(size);

    auto p_in = in.column<ARG>();
    auto in_b = in.bitmask();

    auto fill_value = scalar.value<ARG>();

    for (auto idx = 0; idx < size; ++idx) {
      if (in_b.get(idx))
        out[idx] = p_in[idx];
      else
        out[idx] = fill_value;
    }
    return ColumnView(TYPE_CODE, out, size);
  }

  template <TypeCode TYPE_CODE, std::enable_if_t<TYPE_CODE == TypeCode::STRING> * = nullptr>
  ColumnView operator()(const ColumnView &in, const Scalar &scalar, alloc::Allocator &allocator)
  {
    using ARG = std::string;

    auto size = in.size();

    auto in_o       = in.child(0).column<int32_t>();
    auto in_c       = in.child(1).column<int8_t>();
    auto in_b       = in.bitmask();
    auto null_count = in_b.count_unset_bits();

    auto fill_value = scalar.value<ARG>();

    auto out_o       = allocator.allocate_elements<int32_t>(size + 1);
    size_t num_chars = in.child(1).size() + null_count * fill_value.size();
    auto out_c       = allocator.allocate_elements<int8_t>(num_chars);

    int32_t curr_offset = 0;
    for (auto idx = 0; idx < size; ++idx) {
      bool valid = in_b.get(idx);
      out_o[idx] = curr_offset;
      if (valid) {
        int32_t len = in_o[idx + 1] - in_o[idx];
        memcpy(&out_c[curr_offset], &in_c[in_o[idx]], len);
        curr_offset += len;
      } else {
        int32_t len = fill_value.size();
        memcpy(&out_c[curr_offset], fill_value.c_str(), len);
        curr_offset += len;
      }
    }
    out_o[size] = curr_offset;

    return ColumnView{
      TYPE_CODE,
      nullptr,
      size,
      nullptr,
      {ColumnView{TypeCode::INT32, out_o, size + 1}, ColumnView{TypeCode::INT8, out_c, num_chars}}};
  }

  template <TypeCode TYPE_CODE,
            std::enable_if_t<!(is_primitive_type<TYPE_CODE>::value ||
                               TYPE_CODE == TypeCode::STRING)> * = nullptr>
  ColumnView operator()(const ColumnView &in1, const Scalar &scalar, alloc::Allocator &allocator)
  {
    assert(false);
    return ColumnView(TYPE_CODE);
  }
};

ColumnView broadcast_fillna(const ColumnView &in, const Scalar &scalar, alloc::Allocator &allocator)
{
  return type_dispatch(in.code(), BroadcastFillNa{}, in, scalar, allocator);
}

}  // namespace detail

/*static*/ void BroadcastFillNaTask::cpu_variant(const Task *task,
                                                 const std::vector<PhysicalRegion> &regions,
                                                 Context context,
                                                 Runtime *runtime)
{
  Deserializer ctx{task, regions};

  OutputColumn out;
  Column<true> in;
  Scalar scalar;

  deserialize(ctx, out);
  deserialize(ctx, in);
  deserialize(ctx, scalar);

  if (in.empty()) {
    out.make_empty(true);
    return;
  }

  alloc::DeferredBufferAllocator allocator;
  auto result = detail::broadcast_fillna(in.view(), scalar, allocator);
  out.return_from_view(allocator, result);
}

static void __attribute__((constructor)) register_tasks(void)
{
  BroadcastFillNaTask::register_variants();
}

}  // namespace transform
}  // namespace pandas
}  // namespace legate

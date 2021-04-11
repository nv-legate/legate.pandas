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

#include "unaryop/tasks/unary_op.h"
#include "unaryop/unary_op.h"
#include "column/column.h"
#include "util/allocator.h"
#include "util/type_dispatch.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace unaryop {

using namespace Legion;
using ColumnView = detail::Column;

namespace detail {

namespace numeric {

template <UnaryOpCode OP_CODE, TypeCode TYPE_CODE>
struct UnaryOpImpl {
  ColumnView operator()(const ColumnView &in, alloc::Allocator &allocator)
  {
    using ARG = pandas_type_of<TYPE_CODE>;
    using OP  = UnaryOp<OP_CODE, ARG>;
    using RES = std::result_of_t<OP(ARG)>;

    OP op{};
    auto res_code = pandas_type_code_of<RES>;

    auto size  = in.size();
    auto p_out = allocator.allocate_elements<RES>(size);
    auto p_in  = in.column<ARG>();

    if (!in.nullable()) {
      for (auto idx = 0; idx < size; ++idx) p_out[idx] = op(p_in[idx]);
    } else {
      auto in_b = in.bitmask();
      for (auto idx = 0; idx < size; ++idx) {
        bool valid = in_b.get(idx);
        if (valid) p_out[idx] = op(p_in[idx]);
      }
    }
    return ColumnView(res_code, p_out, size);
  }
};

}  // namespace numeric

struct UnaryOpTypeDispatch {
  template <TypeCode TYPE_CODE>
  ColumnView operator()(UnaryOpCode op_code, const ColumnView &in, alloc::Allocator &allocator)
  {
    switch (op_code) {
      case UnaryOpCode::ABS: {
        return numeric::UnaryOpImpl<UnaryOpCode::ABS, TYPE_CODE>{}(in, allocator);
      }
      case UnaryOpCode::BIT_INVERT: {
        return numeric::UnaryOpImpl<UnaryOpCode::BIT_INVERT, TYPE_CODE>{}(in, allocator);
      }
    }
    assert(false);
    return ColumnView();
  }
};

ColumnView unary_op(UnaryOpCode op_code, const ColumnView &in, alloc::Allocator &allocator)
{
  return type_dispatch_numeric_only(in.code(), UnaryOpTypeDispatch{}, op_code, in, allocator);
}

}  // namespace detail

/*static*/ void UnaryOpTask::cpu_variant(const Task *task,
                                         const std::vector<PhysicalRegion> &regions,
                                         Context context,
                                         Runtime *runtime)
{
  Deserializer ctx{task, regions};

  UnaryOpCode op_code;
  OutputColumn out;
  Column<true> in;

  deserialize(ctx, op_code);
  deserialize(ctx, out);
  deserialize(ctx, in);

  alloc::DeferredBufferAllocator allocator;
  auto result = detail::unary_op(op_code, in.view(), allocator);
  out.return_from_view(allocator, result);
}

static void __attribute__((constructor)) register_tasks(void) { UnaryOpTask::register_variants(); }

}  // namespace unaryop
}  // namespace pandas
}  // namespace legate

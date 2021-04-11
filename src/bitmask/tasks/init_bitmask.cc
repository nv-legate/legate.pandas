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

#include <cmath>

#include "bitmask/tasks/init_bitmask.h"
#include "column/column.h"
#include "column/detail/column.h"
#include "scalar/scalar.h"
#include "util/allocator.h"
#include "util/type_dispatch.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace bitmask {

using namespace Legion;
using ColumnView = detail::Column;

namespace detail {

template <class T>
struct IsValid {
  template <class _T = T, std::enable_if_t<std::is_integral<_T>::value> * = nullptr>
  constexpr Bitmask::AllocType operator()(const T &v, const T &null_value) const
  {
    return static_cast<Bitmask::AllocType>(v != null_value);
  }

  template <class _T = T, std::enable_if_t<!std::is_integral<_T>::value> * = nullptr>
  constexpr Bitmask::AllocType operator()(const T &v, const T &null_value) const
  {
    return static_cast<Bitmask::AllocType>(!std::isnan(v));
  }
};

struct Initializer {
  template <TypeCode CODE>
  ColumnView operator()(const ColumnView &input,
                        const Scalar &null_value,
                        alloc::Allocator &allocator)
  {
    using VAL = pandas_type_of<CODE>;

    auto size      = input.size();
    auto p_bitmask = allocator.allocate_elements<Bitmask::AllocType>(size);

    auto p_input = input.column<VAL>();
    auto val     = null_value.value<VAL>();

    IsValid<VAL> is_valid;
    for (size_t idx = 0; idx < size; ++idx) p_bitmask[idx] = is_valid(p_input[idx], val);

    return ColumnView(pandas_type_code_of<Bitmask::AllocType>, p_bitmask, size);
  }
};

ColumnView initialize_bitmask(const ColumnView &input,
                              const Scalar &null_value,
                              alloc::Allocator &allocator)
{
  return type_dispatch_numeric_only(input.code(), Initializer{}, input, null_value, allocator);
}

}  // namespace detail

/*static*/ void InitBitmaskTask::cpu_variant(const Task *task,
                                             const std::vector<PhysicalRegion> &regions,
                                             Context context,
                                             Runtime *runtime)
{
  Deserializer ctx{task, regions};

  Scalar null_value;
  OutputColumn bitmask;
  Column<true> input;
  deserialize(ctx, null_value);
  deserialize(ctx, bitmask);
  deserialize(ctx, input);

  if (input.empty()) {
    bitmask.make_empty();
    return;
  }

  alloc::DeferredBufferAllocator allocator;
  auto result = detail::initialize_bitmask(input.view(), null_value, allocator);
  bitmask.return_from_view(allocator, result);
}

static void __attribute__((constructor)) register_tasks(void)
{
  InitBitmaskTask::register_variants();
}

}  // namespace bitmask
}  // namespace pandas
}  // namespace legate

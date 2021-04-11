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

#include "index/tasks/find_bounds.h"
#include "index/search.h"
#include "column/column.h"
#include "scalar/scalar.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace index {

using namespace Legion;
using ColumnView = detail::Column;

namespace detail {

coord_t search_forward(const ColumnView &in,
                       const Scalar &to_find,
                       const Rect<1> &bounds,
                       int32_t total_volume)
{
  auto result = search(in, to_find, true);
  if (!result.valid())
    return total_volume;
  else
    return result.value<int64_t>() + bounds.lo[0];
}

coord_t search_backward(const ColumnView &in, const Scalar &to_find, const Rect<1> &bounds)
{
  auto result = search(in, to_find, false);
  if (!result.valid())
    return -1;
  else
    return result.value<int64_t>() + bounds.lo[0];
}

}  // namespace detail

/*static*/ Rect<1> FindBoundsTask::cpu_variant(const Task *task,
                                               const std::vector<PhysicalRegion> &regions,
                                               Context context,
                                               Runtime *runtime)
{
  Deserializer ctx{task, regions};

  Scalar start;
  Scalar stop;
  FromFuture<int64_t> volume;
  Column<true> column;

  deserialize(ctx, start);
  deserialize(ctx, stop);
  deserialize(ctx, volume);
  deserialize(ctx, column);

  const auto &bounds = column.shape();

  auto lo =
    start.valid() ? detail::search_forward(column.view(), start, bounds, volume.value()) : 0;
  auto hi =
    stop.valid() ? detail::search_backward(column.view(), stop, bounds) : volume.value() - 1;

  return Rect<1>{Point<1>{lo}, Point<1>{hi}};
}

static void __attribute__((constructor)) register_tasks(void)
{
  FindBoundsTask::register_variants_with_return<Rect<1>, Rect<1>>();
}

}  // namespace index
}  // namespace pandas
}  // namespace legate

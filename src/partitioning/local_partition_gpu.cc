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

#include "partitioning/local_partition.h"
#include "partitioning/local_partition_args.h"

#include "cudf_util/allocators.h"
#include "cudf_util/column.h"
#include "util/gpu_task_context.h"

#include <cudf/null_mask.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/column/column.hpp>

namespace legate {
namespace pandas {
namespace partition {

using namespace Legion;

using CudfColumns = std::vector<cudf::column_view>;

/*static*/ int64_t LocalPartitionTask::gpu_variant(const Task *task,
                                                   const std::vector<PhysicalRegion> &regions,
                                                   Context context,
                                                   Runtime *runtime)
{
  Deserializer ctx{task, regions};

  detail::LocalPartitionArgs args;
  deserialize(ctx, args);

  int64_t size = static_cast<int64_t>(args.input[0].num_elements());
  if (0 == size) {
    for (auto &column : args.output) column.make_empty(true);

    const coord_t lo = args.input[0].shape().lo[0];
    const coord_t y  = args.hist_rect.lo[1];
    for (coord_t i = 0; i < args.num_pieces; ++i)
      args.hist_acc[Point<2>(i, y)] = Rect<1>(lo, lo - 1);
    return 0;
  }

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  CudfColumns columns;
  for (auto const &column : args.input) columns.push_back(to_cudf_column(column, stream));

  DeferredBufferAllocator mr;
  cudf::table_view table{std::move(columns)};

  auto result = cudf::hash_partition(
    table, args.key_indices, args.num_pieces, cudf::hash_id::HASH_MURMUR3, 12345, stream, &mr);
  from_cudf_table(args.output, std::move(result.first), stream, mr);

  std::vector<int32_t> &offsets = result.second;
  offsets.push_back(args.input[0].num_elements());

  const coord_t lo = args.input[0].shape().lo[0];
  const coord_t y  = args.hist_rect.lo[1];
  for (coord_t i = 0; i < args.num_pieces; ++i)
    args.hist_acc[Point<2>(i, y)] = Rect<1>(lo + offsets[i], lo + offsets[i + 1] - 1);

  return size;
}

}  // namespace partition
}  // namespace pandas
}  // namespace legate

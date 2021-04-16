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

#include <numeric>

#include "partitioning/global_partition.h"

#include "cudf_util/allocators.h"
#include "cudf_util/column.h"
#include "nccl/shuffle.h"
#include "util/cuda_helper.h"
#include "util/gpu_task_context.h"

#include <cudf/partitioning.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

namespace legate {
namespace pandas {
namespace partition {

using namespace Legion;

using CudfColumns = std::vector<cudf::column_view>;

namespace detail {

struct GlobalPartitionArgs {
  ~GlobalPartitionArgs(void) { cleanup(); }
  void cleanup(void);
  void sanity_check(void);

  uint32_t num_pieces;
  std::vector<int32_t> key_indices;
  std::vector<Column<true>> input;
  std::vector<OutputColumn> output;
  ncclComm_t *comm;

  friend void deserialize(Deserializer &ctx, GlobalPartitionArgs &args);
};

void GlobalPartitionArgs::sanity_check(void)
{
  for (auto &column : input) assert(input[0].shape() == column.shape());
}

void GlobalPartitionArgs::cleanup(void)
{
  for (auto &column : input) column.destroy();
  for (auto &column : output) column.destroy();
}

void deserialize(Deserializer &ctx, GlobalPartitionArgs &args)
{
  deserialize(ctx, args.num_pieces);
  deserialize(ctx, args.key_indices);
  deserialize(ctx, args.input);
  deserialize(ctx, args.output);
  deserialize_from_future(ctx, args.comm);
#ifdef DEBUG_PANDAS
  args.sanity_check();
#endif
}

}  // namespace detail

/*static*/ int64_t GlobalPartitionTask::gpu_variant(const Task *task,
                                                    const std::vector<PhysicalRegion> &regions,
                                                    Context context,
                                                    Runtime *runtime)
{
  // Parse arguments
  Deserializer ctx{task, regions};

  detail::GlobalPartitionArgs args;
  detail::deserialize(ctx, args);

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  // First, hash partition the input table
  cudf::table_view table = to_cudf_table(args.input, stream);

  auto num_pieces  = args.num_pieces;
  auto mr          = rmm::mr::get_current_device_resource();
  auto partitioned = cudf::hash_partition(
    table, args.key_indices, args.num_pieces, cudf::hash_id::HASH_MURMUR3, 12345, stream, mr);

  // XXX: It's important to use the number of output columns, instead of the column count of
  //      'partitioned', because the input may contain extra columns just for the purpose of
  //      hash partitioning.  For example, the frontend may have converted categorical
  //      columns into strings and use the latter for the partitioning instead,
  //      for joins on categorical columns having different dictionaries.
  std::vector<int32_t> column_indices_to_send(args.output.size());
  std::iota(column_indices_to_send.begin(), column_indices_to_send.end(), 0);
  auto partitioned_view = partitioned.first->view().select(column_indices_to_send);

  // Before packing subtables into buffers, convert categorical columns to integer codes,
  // as their dictionaries don't need to be transferred (and cuDF doesn't support them
  // in contiguous_split anyway).
  auto converted = comm::extract_dictionaries(partitioned_view);

  // Pack subtables to contiguous buffers for all-to-all exchange
  auto &table_to_send = converted.first;
  std::vector<cudf::size_type> splits(num_pieces - 1, 0);
  if (!partitioned.second.empty())
    std::copy(partitioned.second.begin() + 1, partitioned.second.end(), splits.begin());

  coord_t task_id = task->index_point[0];
  DeferredBufferAllocator output_mr;
  auto result      = comm::shuffle(table_to_send, splits, task_id, args.comm, stream, &output_mr);
  auto result_size = static_cast<int64_t>(result->num_rows());

  // Finally, bind the result to output columns
  auto recovered = comm::embed_dictionaries(std::move(result), converted.second);
  from_cudf_table(args.output, std::move(recovered), stream, output_mr);

  // Make sure all computation is done before we clean up temporary allocations
  // (which will happen when exiting this function)
  SYNC_AND_CHECK_STREAM(stream);

  return result_size;
}

static void __attribute__((constructor)) register_tasks(void)
{
  GlobalPartitionTask::register_variants_with_return<int64_t, int64_t>();
}

}  // namespace partition
}  // namespace pandas
}  // namespace legate

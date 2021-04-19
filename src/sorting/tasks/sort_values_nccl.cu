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

#include "sorting/tasks/sort_values_nccl.h"
#include "column/column.h"
#include "cudf_util/bitmask.h"
#include "cudf_util/column.h"
#include "nccl/util.h"
#include "nccl/shuffle.h"
#include "util/cuda_helper.h"
#include "util/gpu_task_context.h"
#include "util/zip_for_each.h"
#include "deserializer.h"

#include <nccl.h>

#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

namespace legate {
namespace pandas {
namespace sorting {

using namespace Legion;

namespace detail {

// Task arguments
struct SortValuesTaskArgs {
  void sanity_check(void);

  int64_t volume;
  int32_t num_pieces;
  bool put_null_first;
  std::vector<bool> ascending;
  std::vector<int32_t> key_indices;
  std::vector<Column<true>> input;
  std::vector<OutputColumn> output;
  ncclComm_t *comm;

  friend void deserialize(Deserializer &ctx, SortValuesTaskArgs &args);
};

void SortValuesTaskArgs::sanity_check(void)
{
  for (auto &column : input) assert(input[0].shape() == column.shape());
}

void deserialize(Deserializer &ctx, SortValuesTaskArgs &args)
{
  deserialize_from_future(ctx, args.volume);
  deserialize(ctx, args.num_pieces);
  deserialize(ctx, args.put_null_first);
  uint32_t num_key_columns = 0;
  deserialize(ctx, num_key_columns);
  args.ascending.resize(num_key_columns);
  deserialize(ctx, args.ascending, false);
  args.key_indices.resize(num_key_columns);
  deserialize(ctx, args.key_indices, false);

  uint32_t num_columns = 0;
  deserialize(ctx, num_columns);
  args.input.resize(num_columns);
  args.output.resize(num_columns);
  deserialize(ctx, args.input, false);
  deserialize(ctx, args.output, false);
  deserialize_from_future(ctx, args.comm);
}

std::unique_ptr<cudf::table> gather(const cudf::table_view &input,
                                    const std::vector<int32_t> &indices,
                                    cudaStream_t stream,
                                    rmm::mr::device_memory_resource *mr)
{
  DeferredBuffer<int32_t, 1> device_indices_buf{Memory::Z_COPY_MEM, Rect<1>{0, indices.size() - 1}};
  auto device_indices = device_indices_buf.ptr(0);
  for (auto idx = 0; idx < indices.size(); ++idx) device_indices[idx] = indices[idx];

  cudf::column_view gather_map(cudf::data_type(cudf::type_id::INT32),
                               static_cast<cudf::size_type>(indices.size()),
                               device_indices);

  return cudf::detail::gather(input,
                              gather_map,
                              cudf::out_of_bounds_policy::DONT_CHECK,
                              cudf::detail::negative_index_policy::NOT_ALLOWED,
                              stream,
                              mr);
}

// Arguments to the sorting functions
struct SortArgs {
  SortArgs(const Task *task, SortValuesTaskArgs &args);
  ~SortArgs() = default;

  constexpr cudaStream_t stream() const { return gpu_ctx.stream(); }

  GPUTaskContext gpu_ctx;
  cudf::table_view input;
  std::unordered_map<uint32_t, cudf::column_view> dictionaries;
  std::vector<int32_t> key_indices;
  std::vector<cudf::order> column_order;
  std::vector<cudf::null_order> null_precedence;
  int64_t task_id;
  int64_t volume;
  int32_t num_pieces;
  ncclComm_t *comm;
  rmm::mr::device_memory_resource *temp_mr;
  std::unique_ptr<DeferredBufferAllocator> output_mr;
};

SortArgs::SortArgs(const Task *task, SortValuesTaskArgs &args)
  : key_indices(std::move(args.key_indices)),
    task_id(task->index_point[0]),
    volume(args.volume),
    num_pieces(args.num_pieces),
    comm(args.comm),
    output_mr(new DeferredBufferAllocator())
{
  temp_mr = rmm::mr::get_current_device_resource();

  auto input_table              = to_cudf_table(args.input, stream());
  auto converted                = comm::extract_dictionaries(input_table);
  std::tie(input, dictionaries) = std::move(converted);

  for (auto asc : args.ascending) {
    column_order.push_back(asc ? cudf::order::ASCENDING : cudf::order::DESCENDING);
    null_precedence.push_back(asc == args.put_null_first ? cudf::null_order::BEFORE
                                                         : cudf::null_order::AFTER);
  }
}

bool use_sample_sort(SortArgs &args)
{
#ifdef FORCE_SAMPLE_SORT
  return true;
#else
  // Use sample sort only when the average sampling rate is lower than 25%.
  // TODO: Make this magic numbers configurable
  return args.volume / args.num_pieces / 32 >= 4;
#endif
}

std::unique_ptr<cudf::table> all_gather_sort(SortArgs &args)
{
  // Gather all rows and sort them
  auto all_rows = comm::all_gather(
    args.input, args.task_id, args.num_pieces, args.comm, args.stream(), args.temp_mr);

  auto all_rows_view = all_rows->view();
  auto all_rows_keys = all_rows_view.select(args.key_indices);

  auto sorted = cudf::detail::sort_by_key(all_rows_view,
                                          all_rows_keys,
                                          args.column_order,
                                          args.null_precedence,
                                          args.stream(),
                                          args.temp_mr);

  auto start_idx = static_cast<cudf::size_type>(args.volume * args.task_id / args.num_pieces);
  auto stop_idx  = static_cast<cudf::size_type>(args.volume * (args.task_id + 1) / args.num_pieces);

  std::vector<cudf::size_type> indices{start_idx, stop_idx};

  auto sliced = cudf::slice(sorted->view(), indices);

  return std::make_unique<cudf::table>(sliced[0], args.stream(), args.output_mr.get());
}

std::unique_ptr<cudf::table> sample_sort(SortArgs &args)
{
  // Sort the table locally
  auto input_keys     = args.input.select(args.key_indices);
  auto locally_sorted = cudf::detail::sort_by_key(
    args.input, input_keys, args.column_order, args.null_precedence, args.stream(), args.temp_mr);
  auto locally_sorted_keys = locally_sorted->view().select(args.key_indices);

  // Randomly sample keys
  auto num_samples = std::min(32, locally_sorted_keys.num_rows());
  auto samples     = cudf::detail::sample(locally_sorted_keys,
                                      num_samples,
                                      cudf::sample_with_replacement::FALSE,
                                      Realm::Clock::current_time_in_nanoseconds(),
                                      args.stream(),
                                      args.temp_mr);

  // Gather all samples and sort them
  auto all_samples = comm::all_gather(
    samples->view(), args.task_id, args.num_pieces, args.comm, args.stream(), args.temp_mr);

  auto sorted_samples = cudf::detail::sort_by_key(all_samples->view(),
                                                  all_samples->view(),
                                                  args.column_order,
                                                  args.null_precedence,
                                                  args.stream(),
                                                  args.temp_mr);

  // Sample again, but deterministically this time so that tasks agree on the split points they
  // choose
  auto stride = sorted_samples->num_rows() / args.num_pieces;
  std::vector<int32_t> boundary_indices;
  for (auto idx = 0; idx < args.num_pieces - 1; ++idx)
    boundary_indices.push_back((idx + 1) * stride);

  auto dividers =
    detail::gather(sorted_samples->view(), boundary_indices, args.stream(), args.temp_mr);

  // Find split points using the samples
  auto device_splits = cudf::detail::lower_bound(locally_sorted_keys,
                                                 dividers->view(),
                                                 args.column_order,
                                                 args.null_precedence,
                                                 args.stream(),
                                                 args.temp_mr);
  std::vector<int32_t> host_splits(device_splits->size());
  cudaMemcpyAsync(host_splits.data(),
                  device_splits->view().data<int32_t>(),
                  sizeof(int32_t) * device_splits->size(),
                  cudaMemcpyDeviceToHost,
                  args.stream());

  // We should wait until the splits are copied
  SYNC_AND_CHECK_STREAM(args.stream());

  // All-to-all exchange tables
  return comm::shuffle(locally_sorted->view(),
                       host_splits,
                       args.task_id,
                       args.comm,
                       args.stream(),
                       args.output_mr.get());
}

std::unique_ptr<cudf::table> sort(SortArgs &args)
{
  if (use_sample_sort(args))
    return sample_sort(args);
  else
    return all_gather_sort(args);
}

}  // namespace detail

/*static*/ int64_t SortValuesNCCLTask::gpu_variant(const Task *task,
                                                   const std::vector<PhysicalRegion> &regions,
                                                   Context context,
                                                   Runtime *runtime)
{
  Deserializer ctx{task, regions};

  detail::SortValuesTaskArgs task_args;
  detail::deserialize(ctx, task_args);

  detail::SortArgs args(task, task_args);

  auto result      = detail::sort(args);
  auto result_size = result->num_rows();
  auto converted   = comm::embed_dictionaries(std::move(result), args.dictionaries);

  from_cudf_table(task_args.output, std::move(converted), args.stream(), *args.output_mr);

  return result_size;
}

static void __attribute__((constructor)) register_tasks(void)
{
  SortValuesNCCLTask::register_variants_with_return<int64_t, int64_t>();
}

}  // namespace sorting
}  // namespace pandas
}  // namespace legate

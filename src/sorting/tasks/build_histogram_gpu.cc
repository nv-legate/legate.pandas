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

#include "sorting/tasks/build_histogram.h"
#include "cudf_util/column.h"
#include "util/cuda_helper.h"
#include "util/gpu_task_context.h"

#include <cudf/detail/gather.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

namespace legate {
namespace pandas {
namespace sorting {

using namespace Legion;

using Table              = BuildHistogramTask::BuildHistogramArgs::Table;
using BuildHistogramArgs = BuildHistogramTask::BuildHistogramArgs;

/*static*/ void BuildHistogramTask::gpu_variant(const Task *task,
                                                const std::vector<PhysicalRegion> &regions,
                                                Context context,
                                                Runtime *runtime)
{
  Deserializer ctx{task, regions};

  BuildHistogramArgs args;
  deserialize(ctx, args);

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  auto num_keys = args.input.size();

  std::vector<cudf::column_view> input_samples;
  for (auto &column : args.samples) input_samples.push_back(to_cudf_column(column, stream));

  // Sort the samples
  std::vector<cudf::order> column_order;
  for (auto asc : args.ascending)
    column_order.push_back(asc ? cudf::order::ASCENDING : cudf::order::DESCENDING);
  std::vector<cudf::null_order> null_precedence(
    args.ascending.size(),
    args.put_null_first ? cudf::null_order::BEFORE : cudf::null_order::AFTER);
  auto sorted_samples = cudf::detail::sort_by_key(cudf::table_view{input_samples},
                                                  cudf::table_view{input_samples},
                                                  column_order,
                                                  null_precedence,
                                                  stream);
  // Select N - 1 samples
  auto num_samples    = args.samples[0].num_elements();
  auto num_boundaries = args.num_pieces - 1;
  auto stride         = num_samples / args.num_pieces;
  std::vector<int32_t> host_sample_indices;
  for (auto idx = 0; idx < num_boundaries; ++idx) host_sample_indices.push_back((idx + 1) * stride);
  thrust::device_vector<int32_t> sample_indices{host_sample_indices};

  cudf::column_view gather_map{cudf::data_type{cudf::type_id::INT32},
                               static_cast<cudf::size_type>(num_boundaries),
                               thrust::raw_pointer_cast(&sample_indices[0]),
                               nullptr,
                               0};

  auto dividers = cudf::detail::gather(sorted_samples->view(),
                                       gather_map,
                                       cudf::out_of_bounds_policy::DONT_CHECK,
                                       cudf::detail::negative_index_policy::NOT_ALLOWED,
                                       stream);

  std::vector<cudf::column_view> keys;
  for (auto &column : args.input) keys.push_back(to_cudf_column(column, stream));

  auto splits =
    cudf::detail::lower_bound(cudf::table_view{keys}, dividers->view(), column_order, {}, stream);
  std::vector<int32_t> offsets(args.num_pieces + 1);
  offsets[0]               = 0;
  offsets[args.num_pieces] = static_cast<int32_t>(args.input[0].num_elements());
  cudaMemcpyAsync(&offsets[1],
                  splits->view().data<int32_t>(),
                  sizeof(int32_t) * (args.num_pieces - 1),
                  cudaMemcpyDeviceToHost,
                  stream);
  SYNC_AND_CHECK_STREAM(stream);

  const coord_t lo = args.input[0].shape().lo[0];
  const coord_t y  = args.hist_rect.lo[1];
  for (coord_t i = 0; i < args.num_pieces; ++i)
    args.hist_acc[Point<2>(i, y)] = Rect<1>(lo + offsets[i], lo + offsets[i + 1] - 1);
}

}  // namespace sorting
}  // namespace pandas
}  // namespace legate

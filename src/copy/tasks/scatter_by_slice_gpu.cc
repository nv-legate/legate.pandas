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

#include "copy/tasks/scatter_by_slice.h"
#include "copy/materialize.cuh"
#include "column/device_column.h"
#include "cudf_util/allocators.h"
#include "cudf_util/scalar.h"
#include "cudf_util/detail.h"
#include "util/gpu_task_context.h"
#include "util/zip_for_each.h"

#include <cudf/detail/scatter.hpp>

namespace legate {
namespace pandas {
namespace copy {

using namespace Legion;

/*static*/ void ScatterBySliceTask::gpu_variant(const Task *task,
                                                const std::vector<PhysicalRegion> &regions,
                                                Context context,
                                                Runtime *runtime)
{
  Deserializer ctx{task, regions};

  ScatterBySliceTaskArgs args;
  deserialize(ctx, args);

#ifdef DEBUG_PANDAS
  assert(args.requests.size() > 0);
#endif

  if (args.requests.front().target.empty()) {
    for (auto &req : args.requests) req.output.make_empty(true);
    return;
  }

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  DeferredBufferAllocator mr;

  const auto &bounds = args.requests.front().target.shape();
  auto range         = bounds.intersection(args.range);

  auto target_begin = range.lo[0] - bounds.lo[0];
  auto target_end   = range.hi[0] - bounds.lo[0];

  std::vector<cudf::column_view> input_columns;
  std::vector<std::unique_ptr<cudf::scalar>> input_scalars;
  std::vector<std::reference_wrapper<const cudf::scalar>> input_scalars_refs;
  std::vector<cudf::column_view> target_columns;

  for (auto &req : args.requests)
    target_columns.push_back(DeviceColumn<true>{req.target}.to_cudf_column(stream));

  if (args.input_is_scalar)
    for (auto &req : args.requests) {
      auto scalar = to_cudf_scalar(req.scalar_input.raw_ptr(), req.scalar_input.code(), stream);
      input_scalars_refs.push_back(std::cref(*scalar));
      input_scalars.push_back(std::move(scalar));
    }
  else
    for (auto &req : args.requests)
      input_columns.push_back(DeviceColumn<true>{req.input}.to_cudf_column(stream));

  if (target_begin > target_end) {
    for (auto idx = 0; idx < args.requests.size(); ++idx) {
      auto &target = target_columns[idx];
      auto &output = args.requests[idx].output;

      auto result = std::make_unique<cudf::column>(target, stream, &mr);
      DeviceOutputColumn{output}.return_from_cudf_column(mr, result->view(), stream);
    }
  } else if (args.input_is_scalar) {
    auto scatter_map = materialize(
      Rect<1>(target_begin, target_end), 0, 1, stream, rmm::mr::get_current_device_resource());
    auto result = cudf::detail::scatter(input_scalars_refs,
                                        scatter_map->view(),
                                        cudf::table_view{std::move(target_columns)},
                                        false,
                                        stream,
                                        &mr);

    auto result_view = result->view();
    util::for_each(args.requests, result_view, [&](auto &req, auto &result) {
      DeviceOutputColumn{req.output}.return_from_cudf_column(mr, result, stream);
    });
  } else {
    for (auto idx = 0; idx < args.requests.size(); ++idx) {
      auto &source = input_columns[idx];
      auto &target = target_columns[idx];
      auto &output = args.requests[idx].output;

      auto result =
        cudf::detail::copy_range(source, target, 0, source.size(), target_begin, stream, &mr);
      DeviceOutputColumn{output}.return_from_cudf_column(mr, result->view(), stream);
    }
  }
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

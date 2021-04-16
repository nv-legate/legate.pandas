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

#include "copy/tasks/scatter_by_mask.h"
#include "cudf_util/allocators.h"
#include "cudf_util/column.h"
#include "cudf_util/scalar.h"
#include "util/gpu_task_context.h"
#include "util/zip_for_each.h"

#include <cudf/detail/scatter.hpp>

namespace legate {
namespace pandas {
namespace copy {

using namespace Legion;

/*static*/ void ScatterByMaskTask::gpu_variant(const Task *task,
                                               const std::vector<PhysicalRegion> &regions,
                                               Context context,
                                               Runtime *runtime)
{
  Deserializer ctx{task, regions};

  ScatterByMaskTaskArgs args;
  deserialize(ctx, args);

  if (args.mask.empty()) {
    for (auto &req : args.requests) req.output.make_empty(true);
    return;
  }

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  DeferredBufferAllocator mr;

  auto mask = to_cudf_column(args.mask, stream);

  std::vector<cudf::column_view> input_columns;
  std::vector<std::unique_ptr<cudf::scalar>> input_scalars;
  std::vector<std::reference_wrapper<const cudf::scalar>> input_scalars_refs;
  std::vector<cudf::column_view> target_columns;

  for (auto &req : args.requests) target_columns.push_back(to_cudf_column(req.target, stream));

  if (args.input_is_scalar)
    for (auto &req : args.requests) {
      auto scalar = to_cudf_scalar(req.scalar_input.raw_ptr(), req.scalar_input.code(), stream);
      input_scalars_refs.push_back(std::cref(*scalar));
      input_scalars.push_back(std::move(scalar));
    }
  else
    for (auto &req : args.requests) input_columns.push_back(to_cudf_column(req.input, stream));

  std::unique_ptr<cudf::table> result;
  if (args.input_is_scalar)
    result = cudf::detail::boolean_mask_scatter(
      input_scalars_refs, cudf::table_view{std::move(target_columns)}, mask, stream, &mr);
  else
    result = cudf::detail::boolean_mask_scatter(cudf::table_view{std::move(input_columns)},
                                                cudf::table_view{std::move(target_columns)},
                                                mask,
                                                stream,
                                                &mr);

  auto result_columns = result->release();
  util::for_each(args.requests, result_columns, [&](auto &req, auto &result) {
    from_cudf_column(req.output, std::move(result), stream, mr);
  });
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

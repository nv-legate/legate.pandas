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

#include <vector>

#include "category/tasks/drop_duplicates.h"
#include "column/column.h"
#include "column/device_column.h"
#include "cudf_util/allocators.h"
#include "util/cuda_helper.h"
#include "util/gpu_task_context.h"
#include "deserializer.h"

#include <cudf/copying.hpp>
#include <cudf/detail/concatenate.cuh>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>

namespace legate {
namespace pandas {
namespace category {

/*static*/ void DropDuplicatesTask::gpu_variant(const Legion::Task *task,
                                                const std::vector<Legion::PhysicalRegion> &regions,
                                                Legion::Context context,
                                                Legion::Runtime *runtime)
{
  Deserializer ctx{task, regions};

  uint32_t num_inputs = 0;
  deserialize(ctx, num_inputs);

  OutputColumn out;
  deserialize(ctx, out);

  std::vector<Column<true>> inputs;
  for (uint32_t i = 0; i < num_inputs; ++i) {
    Column<true> in;
    deserialize(ctx, in);
    if (!in.valid()) break;
    inputs.push_back(in);
  }

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  std::vector<cudf::column_view> columns;
  for (auto const &input : inputs)
    columns.push_back(DeviceColumn<true>{input}.to_cudf_column(stream));

  DeferredBufferAllocator mr;
  auto input = cudf::detail::concatenate(columns, stream, &mr);

  auto result_columns = cudf::detail::drop_duplicates(cudf::table_view{{input->view()}},
                                                      std::vector<cudf::size_type>{0},
                                                      cudf::duplicate_keep_option::KEEP_FIRST,
                                                      cudf::null_equality::EQUAL,
                                                      stream,
                                                      &mr)
                          ->release();
  std::unique_ptr<cudf::column> result_column(std::move(result_columns.front()));

  if (input->has_nulls()) {
    result_column = std::make_unique<cudf::column>(
      cudf::slice(result_column->view(), std::vector<cudf::size_type>{1, result_column->size()})
        .front(),
      stream,
      &mr);
  }
  // We need to synchronize here so that we can get the correct output size
  SYNC_AND_CHECK_STREAM(stream);

  cudf::strings_column_view result{result_column->view()};

  size_t size       = static_cast<size_t>(result.size());
  size_t chars_size = static_cast<size_t>(result.chars_size());

  auto &out_offsets = out.child(0);
  auto &out_chars   = out.child(1);

  out.allocate(size);
  out_offsets.allocate(size > 0 ? size + 1 : 0);
  out_chars.allocate(chars_size);

  if (size == 0) return;

  cudaMemcpyAsync(out_offsets.raw_column<int32_t>(),
                  result.offsets().data<int32_t>(),
                  (size + 1) * sizeof(int32_t),
                  cudaMemcpyDeviceToDevice,
                  stream);
  cudaMemcpyAsync(out_chars.raw_column<int8_t>(),
                  result.chars().data<int8_t>(),
                  chars_size,
                  cudaMemcpyDeviceToDevice,
                  stream);
}

}  // namespace category
}  // namespace pandas
}  // namespace legate

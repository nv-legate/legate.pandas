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

#include <unordered_set>

#include "merge/merge.h"
#include "column/device_column.h"
#include "cudf_util/allocators.h"
#include "util/gpu_task_context.h"
#include "util/zip_for_each.h"

#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/replace.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/table/table.hpp>
#include <cudf/join.hpp>

namespace legate {
namespace pandas {
namespace merge {

using OutputColumns = std::vector<std::pair<bool, DeviceOutputColumn *>>;

namespace detail {

cudf::table_view to_cudf_table(const std::vector<Column<true>> &columns, cudaStream_t stream)
{
  std::vector<cudf::column_view> column_views;
  for (auto &column : columns)
    column_views.push_back(DeviceColumn<true>{column}.to_cudf_column(stream));
  return cudf::table_view(column_views);
}

}  // namespace detail

/*static*/ int64_t MergeTask::gpu_variant(const Legion::Task *task,
                                          const std::vector<Legion::PhysicalRegion> &regions,
                                          Legion::Context context,
                                          Legion::Runtime *runtime)
{
  Deserializer ctx(task, regions);

  MergeArgs args;
  deserialize(ctx, args);

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  auto left_input  = detail::to_cudf_table(args.left_input, stream);
  auto right_input = detail::to_cudf_table(args.right_input, stream);

  DeferredBufferAllocator mr;

  auto matched = cudf::dictionary::detail::match_dictionaries(
    {left_input.select(args.left_on), right_input.select(args.right_on)}, stream, &mr);

  left_input  = scatter_columns(matched.second.front(), args.left_on, left_input);
  right_input = scatter_columns(matched.second.back(), args.right_on, right_input);

  auto left_keys  = left_input.select(args.left_on);
  auto right_keys = right_input.select(args.right_on);

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_indexer;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> right_indexer;
  cudf::hash_join joiner(right_keys, cudf::null_equality::EQUAL, stream);

  switch (args.join_type) {
    case JoinTypeCode::INNER: {
      std::tie(left_indexer, right_indexer) =
        joiner.inner_join(left_keys, cudf::null_equality::EQUAL, stream, &mr);
      break;
    }
    case JoinTypeCode::LEFT: {
      std::tie(left_indexer, right_indexer) =
        joiner.left_join(left_keys, cudf::null_equality::EQUAL, stream, &mr);
      break;
    }
    case JoinTypeCode::OUTER: {
      std::tie(left_indexer, right_indexer) =
        joiner.full_join(left_keys, cudf::null_equality::EQUAL, stream, &mr);
      break;
    }
    default: {
      assert(false);
      break;
    }
  }

  // Return dictionary columns using inputs, as they may not appear in the output when the output is
  // empty.
  auto left_gather_target  = left_input.select(args.left_indices());
  auto right_gather_target = right_input.select(args.right_indices());

  auto return_dictionary = [&](auto &output, auto &input) {
    if (input.type().id() == cudf::type_id::DICTIONARY32 && output.num_children() > 1) {
      if (task->index_point[0] == 0)
        DeviceOutputColumn(output.child(1)).return_from_cudf_column(mr, input.child(1), stream);
      else
        output.child(1).make_empty();
    }
  };
  util::for_each(args.left_output, left_gather_target, return_dictionary);
  util::for_each(args.right_output, right_gather_target, return_dictionary);

  auto out_of_bounds_policy = args.join_type == JoinTypeCode::INNER
                                ? cudf::out_of_bounds_policy::DONT_CHECK
                                : cudf::out_of_bounds_policy::NULLIFY;

  // If this is not an outer join, simply construct outputs by gathering respective inputs
  std::unique_ptr<cudf::table> left_output;
  std::unique_ptr<cudf::table> right_output;
  std::vector<std::unique_ptr<cudf::column>> temp_columns;

  cudf::table_view left_output_view;
  cudf::table_view right_output_view;
  if (args.join_type != JoinTypeCode::OUTER) {
    left_output  = cudf::detail::gather(left_gather_target,
                                       left_indexer->begin(),
                                       left_indexer->end(),
                                       out_of_bounds_policy,
                                       stream,
                                       &mr);
    right_output = cudf::detail::gather(right_gather_target,
                                        right_indexer->begin(),
                                        right_indexer->end(),
                                        out_of_bounds_policy,
                                        stream,
                                        &mr);

    left_output_view  = left_output->view();
    right_output_view = right_output->view();
  }
  // For an outer join, we need to combine keys from both inputs
  else {
    left_output = cudf::detail::gather(
      left_input, left_indexer->begin(), left_indexer->end(), out_of_bounds_policy, stream, &mr);
    right_output = cudf::detail::gather(
      right_input, right_indexer->begin(), right_indexer->end(), out_of_bounds_policy, stream, &mr);

    std::vector<int32_t> left_common_indices;
    std::vector<int32_t> right_common_indices;
    for (auto &pair : args.common_columns) {
      left_common_indices.push_back(pair.first);
      right_common_indices.push_back(pair.second);
    }

    auto left_common_keys  = left_output->view().select(left_common_indices);
    auto right_common_keys = right_output->view().select(right_common_indices);

    std::vector<cudf::column_view> updated_keys;

    util::for_each(left_common_keys, right_common_keys, [&](auto &left, auto &right) {
      temp_columns.push_back(cudf::detail::replace_nulls(left, right, stream, &mr));
      updated_keys.push_back(temp_columns.back()->view());
    });

    left_output_view = scatter_columns(
      cudf::table_view(std::move(updated_keys)), left_common_indices, left_output->view());
    right_output_view = right_output->view().select(args.right_indices());
  }

  auto return_column = [&](auto &output, auto &cudf_output) {
    if (cudf_output.type().id() == cudf::type_id::DICTIONARY32) {
      if (cudf_output.size() == 0) {
        output.make_empty(false);
        output.child(0).make_empty();
      } else {
        auto codes = cudf_output.child(0);
        if (codes.type().id() != cudf::type_id::UINT32) {
          temp_columns.push_back(
            cudf::detail::cast(codes, cudf::data_type{cudf::type_id::UINT32}, stream, &mr));
          codes = temp_columns.back()->view();
        }

        cudf::column_view codes_only(cudf_output.type(),
                                     cudf_output.size(),
                                     cudf_output.head(),
                                     cudf_output.null_mask(),
                                     -1,
                                     0,
                                     {codes});
        DeviceOutputColumn(output).return_from_cudf_column(mr, codes_only, stream);
      }
    } else {
      if (cudf_output.size() == 0)
        output.make_empty(true);
      else
        DeviceOutputColumn(output).return_from_cudf_column(mr, cudf_output, stream);
    }
  };

  util::for_each(args.left_output, left_output_view, return_column);
  util::for_each(args.right_output, right_output_view, return_column);

  return left_output->num_rows();
}

}  // namespace merge
}  // namespace pandas
}  // namespace legate

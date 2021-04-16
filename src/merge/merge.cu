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
#include "cudf_util/allocators.h"
#include "cudf_util/column.h"
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

  auto left_input  = to_cudf_table(args.left_input, stream);
  auto right_input = to_cudf_table(args.right_input, stream);

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
      if (task->index_point[0] == 0) {
        auto dict = std::make_unique<cudf::column>(input.child(1), stream, &mr);
        from_cudf_column(output.child(1), std::move(dict), stream, mr);
      } else
        output.child(1).make_empty();
    }
  };
  util::for_each(args.left_output, left_gather_target, return_dictionary);
  util::for_each(args.right_output, right_gather_target, return_dictionary);

  auto out_of_bounds_policy = args.join_type == JoinTypeCode::INNER
                                ? cudf::out_of_bounds_policy::DONT_CHECK
                                : cudf::out_of_bounds_policy::NULLIFY;

  // If this is not an outer join, simply construct outputs by gathering respective inputs
  std::vector<std::unique_ptr<cudf::column>> left_output;
  std::vector<std::unique_ptr<cudf::column>> right_output;
  std::vector<std::unique_ptr<cudf::column>> temp_columns;

  if (args.join_type != JoinTypeCode::OUTER) {
    left_output = cudf::detail::gather(left_gather_target,
                                       left_indexer->begin(),
                                       left_indexer->end(),
                                       out_of_bounds_policy,
                                       stream,
                                       &mr)
                    ->release();
    right_output = cudf::detail::gather(right_gather_target,
                                        right_indexer->begin(),
                                        right_indexer->end(),
                                        out_of_bounds_policy,
                                        stream,
                                        &mr)
                     ->release();
  }
  // For an outer join, we need to combine keys from both inputs
  else {
    auto left_output_tbl = cudf::detail::gather(
      left_input, left_indexer->begin(), left_indexer->end(), out_of_bounds_policy, stream, &mr);
    auto right_output_tbl = cudf::detail::gather(
      right_input, right_indexer->begin(), right_indexer->end(), out_of_bounds_policy, stream, &mr);

    std::vector<int32_t> left_common_indices;
    std::vector<int32_t> right_common_indices;
    for (auto &pair : args.common_columns) {
      left_common_indices.push_back(pair.first);
      right_common_indices.push_back(pair.second);
    }

    auto left_common_keys  = left_output_tbl->view().select(left_common_indices);
    auto right_common_keys = right_output_tbl->view().select(right_common_indices);

    std::vector<std::unique_ptr<cudf::column>> updated_keys;
    util::for_each(left_common_keys, right_common_keys, [&](auto &left, auto &right) {
      updated_keys.push_back(cudf::detail::replace_nulls(left, right, stream, &mr));
    });

    left_output = left_output_tbl->release();
    util::for_each(updated_keys, left_common_indices, [&](auto &updated_key, auto idx) {
      left_output[idx] = std::move(updated_key);
    });

    auto right_output_all = right_output_tbl->release();
    for (auto idx : args.right_indices()) right_output.push_back(std::move(right_output_all[idx]));
  }

  auto return_column = [&](auto &output, auto &cudf_output) {
    if (cudf_output->type().id() == cudf::type_id::DICTIONARY32) {
      if (cudf_output->size() == 0) {
        output.make_empty(false);
        output.child(0).make_empty();
      } else {
        auto size     = cudf_output->size();
        auto contents = cudf_output->release();

        std::vector<std::unique_ptr<cudf::column>> children;

        auto &codes = contents.children[0];
        if (codes->type().id() != cudf::type_id::UINT32)
          children.push_back(
            cudf::detail::cast(codes->view(), cudf::data_type{cudf::type_id::UINT32}, stream, &mr));
        else
          children.push_back(std::move(codes));

        auto codes_only =
          std::make_unique<cudf::column>(cudf::data_type(cudf::type_id::DICTIONARY32),
                                         size,
                                         rmm::device_buffer{},
                                         std::move(*contents.null_mask),
                                         -1,
                                         std::move(children));
        from_cudf_column(output, std::move(codes_only), stream, mr);
      }
    } else {
      if (cudf_output->size() == 0)
        output.make_empty(true);
      else
        from_cudf_column(output, std::move(cudf_output), stream, mr);
    }
  };

  util::for_each(args.left_output, left_output, return_column);
  util::for_each(args.right_output, right_output, return_column);

  return left_indexer->size();
}

}  // namespace merge
}  // namespace pandas
}  // namespace legate

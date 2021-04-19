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
#include <unordered_map>

#include "column/detail/column.h"
#include "copy/concatenate.h"
#include "copy/gather.h"
#include "merge/merge.h"
#include "table/row_wrappers.h"
#include "util/allocator.h"
#include "util/zip_for_each.h"

namespace legate {
namespace pandas {
namespace merge {

using namespace Legion;

using InputTable  = MergeTask::MergeArgs::InputTable;
using OutputTable = MergeTask::MergeArgs::OutputTable;
using ColumnViews = std::vector<detail::Column>;

void MergeTask::MergeArgs::sanity_check(void)
{
  assert(left_on.size() == right_on.size());

  for (auto &column : left_input) assert(left_input[0].num_elements() == column.num_elements());
  for (auto &column : right_input) assert(right_input[0].num_elements() == column.num_elements());

  for (auto idx : left_on) assert(idx >= 0 && idx < left_input.size());
  for (auto idx : right_on) assert(idx >= 0 && idx < right_input.size());
}

namespace detail {

using HashSet   = std::unordered_set<table::Row, table::RowHasher, table::RowEqual>;
using HashTable = std::unordered_multimap<table::Row, int64_t, table::RowHasher, table::RowEqual>;

static void build_table(HashTable &table, ColumnViews &right_keys)
{
  size_t size = right_keys[0].size();
  if (size == 0) return;

  for (size_t i = 0; i < size; ++i) {
    table::Row key{right_keys, i};
    table.insert({key, i});
  }
}

static int64_t probe_table(HashTable &table,
                           HashSet &matched_keys,
                           ColumnViews &left_keys,
                           JoinTypeCode code)
{
  size_t size = left_keys[0].size();

  int64_t total_num_matches = 0;
  for (size_t i = 0; i < size; ++i) {
    table::Row key{left_keys, i};
    auto range = table.equal_range(key);

    size_t num_matches = 0;
    if (range.first != table.end()) {
      for (auto it = range.first; it != range.second; ++it) {
        ++num_matches;
        matched_keys.insert(it->first);
      }
    } else
      num_matches = static_cast<size_t>(code != JoinTypeCode::INNER);

    total_num_matches += num_matches;
  }

  if (code == JoinTypeCode::OUTER)
    for (auto pair : table)
      total_num_matches += matched_keys.find(pair.first) == matched_keys.end();

  return total_num_matches;
}

static void product(HashTable &table,
                    HashSet &matched_keys,
                    ColumnViews &left_keys,
                    std::vector<int64_t> &left_indexer,
                    std::vector<int64_t> &right_indexer,
                    JoinTypeCode code)
{
  size_t size = left_keys[0].size();

  size_t out_idx = 0;

  for (size_t in_idx = 0; in_idx < size; ++in_idx) {
    table::Row key{left_keys, in_idx};
    auto range = table.equal_range(key);
    if (range.first != table.end())
      for (auto it = range.first; it != range.second; ++it) {
        left_indexer[out_idx]    = static_cast<int64_t>(in_idx);
        right_indexer[out_idx++] = static_cast<int64_t>(it->second);
      }
    else if (code != JoinTypeCode::INNER) {
      left_indexer[out_idx]    = static_cast<int64_t>(in_idx);
      right_indexer[out_idx++] = -1;
    }
  }

  if (code == JoinTypeCode::OUTER)
    for (auto pair : table) {
      if (matched_keys.find(pair.first) != matched_keys.end()) continue;
      left_indexer[out_idx]    = -1;
      right_indexer[out_idx++] = pair.second;
    }
  assert(out_idx == left_indexer.size());
}

std::vector<int32_t> generate_indices(uint32_t num_indices,
                                      const std::unordered_set<int32_t> &to_exclude)
{
  std::vector<int32_t> indices;
  for (uint32_t idx = 0; idx < num_indices; ++idx)
    if (to_exclude.find(idx) == to_exclude.end()) indices.push_back(idx);
  return indices;
}

}  // namespace detail

std::vector<int32_t> MergeTask::MergeArgs::left_indices() const
{
  std::unordered_set<int32_t> to_exclude;
  if (!output_common_columns_to_left)
    for (auto &pair : common_columns) to_exclude.insert(pair.first);
  return detail::generate_indices(left_input.size(), to_exclude);
}

std::vector<int32_t> MergeTask::MergeArgs::right_indices() const
{
  std::unordered_set<int32_t> to_exclude;
  if (output_common_columns_to_left)
    for (auto &pair : common_columns) to_exclude.insert(pair.second);
  return detail::generate_indices(right_input.size(), to_exclude);
}

/*static*/ int64_t MergeTask::cpu_variant(const Task *task,
                                          const std::vector<PhysicalRegion> &regions,
                                          Context context,
                                          Runtime *runtime)
{
  Deserializer ctx{task, regions};
  MergeArgs args;
  deserialize(ctx, args);

  ColumnViews left_keys;
  ColumnViews right_keys;

  for (auto idx : args.left_on) left_keys.push_back(args.left_input[idx].view());
  for (auto idx : args.right_on) right_keys.push_back(args.right_input[idx].view());

  detail::HashSet matched_keys;
  detail::HashTable table;
  detail::build_table(table, right_keys);
  int64_t num_elements = detail::probe_table(table, matched_keys, left_keys, args.join_type);

  std::vector<int64_t> left_indexer(num_elements);
  std::vector<int64_t> right_indexer(num_elements);
  detail::product(table, matched_keys, left_keys, left_indexer, right_indexer, args.join_type);

  bool out_of_range = args.join_type != JoinTypeCode::INNER;

  std::unordered_set<int32_t> left_common_indices;
  std::unordered_set<int32_t> right_common_indices;

  for (auto &pair : args.common_columns) {
    left_common_indices.insert(pair.first);
    right_common_indices.insert(pair.second);
  }

  using ColumnRef       = std::reference_wrapper<Column<true>>;
  using OutputColumnRef = std::reference_wrapper<OutputColumn>;

  std::vector<std::pair<OutputColumn, Column<true>>> left_columns;
  std::vector<std::pair<OutputColumn, Column<true>>> right_columns;

  std::vector<std::tuple<OutputColumn, Column<true>, Column<true>>> common_columns;
  common_columns.resize(args.common_columns.size());

  for (int32_t in_idx = 0, out_idx = 0, common_idx = 0; in_idx < args.left_input.size(); ++in_idx) {
    auto &input = args.left_input[in_idx];
    if (left_common_indices.find(in_idx) == left_common_indices.end()) {
      auto &output = args.left_output[out_idx++];
      left_columns.push_back(std::make_pair(std::move(output), std::move(input)));
    } else {
      if (args.output_common_columns_to_left) {
        auto &output                            = args.left_output[out_idx++];
        std::get<0>(common_columns[common_idx]) = std::move(output);
        std::get<1>(common_columns[common_idx]) = std::move(input);
      } else
        std::get<1>(common_columns[common_idx]) = std::move(input);
      ++common_idx;
    }
  }

  for (int32_t in_idx = 0, out_idx = 0, common_idx = 0; in_idx < args.right_input.size();
       ++in_idx) {
    auto &input = args.right_input[in_idx];
    if (right_common_indices.find(in_idx) == right_common_indices.end()) {
      auto &output = args.right_output[out_idx++];
      right_columns.push_back(std::make_pair(std::move(output), std::move(input)));
    } else {
      if (!args.output_common_columns_to_left) {
        auto &output                            = args.right_output[out_idx++];
        std::get<0>(common_columns[common_idx]) = std::move(output);
        std::get<2>(common_columns[common_idx]) = std::move(input);
      } else
        std::get<2>(common_columns[common_idx]) = std::move(input);
      ++common_idx;
    }
  }

  auto maybe_copy_dictionary = [&](auto &output, auto &input) {
    if (input.code() == TypeCode::CAT32 && output.num_children() > 1) {
      if (task->index_point[0] == 0)
        output.child(1).copy(input.child(1));
      else
        output.child(1).make_empty();
    }
  };

  alloc::DeferredBufferAllocator allocator;

  for (auto &pair : left_columns) {
    auto &output  = pair.first;
    auto &input   = pair.second;
    auto gathered = copy::gather(
      input.view(), left_indexer, out_of_range, copy::OutOfRangePolicy::NULLIFY, allocator);
    output.return_from_view(allocator, gathered);
    maybe_copy_dictionary(output, input);
  }

  for (auto &pair : right_columns) {
    auto &output  = pair.first;
    auto &input   = pair.second;
    auto gathered = copy::gather(
      input.view(), right_indexer, out_of_range, copy::OutOfRangePolicy::NULLIFY, allocator);
    output.return_from_view(allocator, gathered);
    maybe_copy_dictionary(output, input);
  }

  if (args.join_type != JoinTypeCode::OUTER) {
    for (auto &tpl : common_columns) {
      auto &output  = std::get<0>(tpl);
      auto &input   = std::get<1>(tpl);
      auto gathered = copy::gather(
        input.view(), left_indexer, out_of_range, copy::OutOfRangePolicy::NULLIFY, allocator);
      output.return_from_view(allocator, gathered);
      maybe_copy_dictionary(output, input);
    }
  } else {
    auto split_offset = left_keys[0].size();
    std::vector<int64_t> left_subindexer(&left_indexer[0], &left_indexer[split_offset]);
    std::vector<int64_t> right_subindexer(&right_indexer[split_offset],
                                          &right_indexer[num_elements]);

    for (auto &tpl : common_columns) {
      auto &output      = std::get<0>(tpl);
      auto &left_input  = std::get<1>(tpl);
      auto &right_input = std::get<2>(tpl);
#ifdef DEBUG_PANDAS
      assert(args.output_common_columns_to_left);
#endif
      auto left_gathered  = copy::gather(left_input.view(),
                                        left_subindexer,
                                        out_of_range,
                                        copy::OutOfRangePolicy::NULLIFY,
                                        allocator);
      auto right_gathered = copy::gather(right_input.view(),
                                         right_subindexer,
                                         out_of_range,
                                         copy::OutOfRangePolicy::NULLIFY,
                                         allocator);
      auto result         = copy::concatenate({left_gathered, right_gathered}, allocator);
      output.return_from_view(allocator, result);
      maybe_copy_dictionary(output, left_input);
    }
  }

  return num_elements;
}

void deserialize(Deserializer &ctx, MergeTask::MergeArgs &args)
{
  deserialize(ctx, args.join_type);
  deserialize(ctx, args.output_common_columns_to_left);

  deserialize(ctx, args.left_on);
  deserialize(ctx, args.right_on);
  deserialize(ctx, args.common_columns);

  deserialize(ctx, args.left_input);
  deserialize(ctx, args.right_input);

  deserialize(ctx, args.left_output);
  deserialize(ctx, args.right_output);

#ifdef DEBUG_PANDAS
  args.sanity_check();
#endif
}

static void __attribute__((constructor)) register_tasks(void)
{
  MergeTask::register_variants_with_return<int64_t, int64_t>();
}

}  // namespace merge
}  // namespace pandas
}  // namespace legate

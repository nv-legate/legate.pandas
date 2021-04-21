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

#include <unordered_map>

#include "copy/tasks/drop_duplicates.h"
#include "column/column.h"
#include "copy/concatenate.h"
#include "copy/gather.h"
#include "table/row_wrappers.h"
#include "util/zip_for_each.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace copy {

using namespace Legion;

using HashTable =
  std::unordered_map<table::Row, std::pair<size_t, size_t>, table::RowHasher, table::RowEqual>;

/*static*/ int64_t DropDuplicatesTask::cpu_variant(const Task *task,
                                                   const std::vector<PhysicalRegion> &regions,
                                                   Context context,
                                                   Runtime *runtime)
{
  Deserializer ctx{task, regions};

  KeepMethod method;
  deserialize(ctx, method);

  std::vector<int32_t> subset;
  deserialize(ctx, subset);

  uint32_t num_inputs{0};
  deserialize(ctx, num_inputs);

  std::vector<std::vector<Column<true>>> input_columns;
  std::vector<OutputColumn> outputs;

  for (auto idx = 0; idx < num_inputs; ++idx) {
    std::vector<Column<true>> columns;
    deserialize(ctx, columns);
    if (!columns[0].valid()) continue;
    input_columns.push_back(std::move(columns));
  }

  deserialize(ctx, outputs);

  alloc::DeferredBufferAllocator allocator{};

  detail::Table input;
  if (input_columns.size() > 1) {
    std::vector<detail::Table> tables;
    for (auto &columns : input_columns) {
      std::vector<detail::Column> views;
      for (auto &column : columns) views.push_back(column.view());
      tables.push_back(detail::Table(std::move(views)));
    }
    input = concatenate(tables, allocator);
  } else {
    std::vector<detail::Column> views;
    for (auto &column : input_columns[0]) views.push_back(column.view());
    input = detail::Table(std::move(views));
  }

  auto size = input.size();
  auto keys = input.select(subset);
  HashTable table{};

  for (size_t idx = 0; idx < size; ++idx) {
    table::Row row{keys, idx};
    auto finder = table.find(row);
    if (finder == table.end())
      table[row] = std::make_pair(idx, idx);
    else {
      auto &p  = finder->second;
      p.first  = std::min(p.first, idx);
      p.second = std::max(p.second, idx);
    }
  }

  std::vector<int64_t> mapping;
  switch (method) {
    case KeepMethod::FIRST: {
      for (auto &pair : table) mapping.push_back(pair.second.first);
      break;
    }
    case KeepMethod::LAST: {
      for (auto &pair : table) mapping.push_back(pair.second.second);
      break;
    }
    case KeepMethod::NONE: {
      for (auto &pair : table) {
        if (pair.second.first != pair.second.second) continue;
        mapping.push_back(pair.second.first);
      }
      break;
    }
  }

  auto to_gather = input.release();
  util::for_each(to_gather, outputs, [&](auto &to_gather, auto &output) {
    auto gathered = gather(to_gather, mapping, false, OutOfRangePolicy::IGNORE, allocator);
    output.return_from_view(allocator, gathered);
  });

  return mapping.size();
}

static void __attribute__((constructor)) register_tasks(void)
{
  DropDuplicatesTask::register_variants_with_return<int64_t, int64_t>();
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

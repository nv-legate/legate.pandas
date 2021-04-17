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

#pragma once

#include "pandas.h"
#include "deserializer.h"
#include "column/column.h"

namespace legate {
namespace pandas {
namespace groupby {

// Task for all group-by reductions
class GroupByReductionTask : public PandasTask<GroupByReductionTask> {
 public:
  struct GroupByArgs {
    void sanity_check(void);

    typedef std::vector<Column<true>> ColumnsRO;
    typedef std::vector<OutputColumn> ColumnsWO;
    std::vector<ColumnsRO> in_keys;
    std::vector<ColumnsRO> in_values;
    std::vector<std::vector<AggregationCode>> all_aggs;
    ColumnsWO out_keys;
    std::vector<ColumnsWO> all_out_values;

    friend void deserialize(Deserializer &ctx, GroupByArgs &args);
  };

 public:
  static const int TASK_ID = OpCode::GROUPBY_REDUCE;

 public:
  static int64_t cpu_variant(const Legion::Task *task,
                             const std::vector<Legion::PhysicalRegion> &regions,
                             Legion::Context context,
                             Legion::Runtime *runtime);
#ifdef LEGATE_USE_CUDA
  static int64_t gpu_variant(const Legion::Task *task,
                             const std::vector<Legion::PhysicalRegion> &regions,
                             Legion::Context context,
                             Legion::Runtime *runtime);
#endif
};

}  // namespace groupby
}  // namespace pandas
}  // namespace legate

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
#include "column/column.h"

namespace legate {
namespace pandas {
namespace merge {

// Count the number of entries in the result of a join
class MergeTask : public PandasTask<MergeTask> {
 public:
  struct MergeArgs {
    void sanity_check(void);

    std::vector<int32_t> left_indices() const;
    std::vector<int32_t> right_indices() const;

    using InputTable  = std::vector<Column<true>>;
    using OutputTable = std::vector<OutputColumn>;

    JoinTypeCode join_type;
    bool output_common_columns_to_left;
    std::vector<int32_t> left_on;
    std::vector<int32_t> right_on;
    std::vector<std::pair<int32_t, int32_t>> common_columns;
    InputTable left_input;
    InputTable right_input;
    OutputTable left_output;
    OutputTable right_output;

    friend void deserialize(Deserializer &ctx, MergeArgs &args);
  };

 public:
  static const int TASK_ID = OpCode::MERGE;

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

}  // namespace merge
}  // namespace pandas
}  // namespace legate

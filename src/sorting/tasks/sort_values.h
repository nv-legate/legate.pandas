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
namespace sorting {

class SortValuesTask : public PandasTask<SortValuesTask> {
 public:
  struct SortValuesArgs {
    ~SortValuesArgs(void) { cleanup(); }
    void cleanup(void);
    void sanity_check(void);

    using InputTable = std::vector<Column<true>>;

    bool use_output_only_columns;
    bool put_null_first;
    std::vector<bool> ascending;
    std::vector<int32_t> key_indices;
    InputTable input;
    std::vector<OutputColumn> output;

    friend void deserialize(Deserializer &ctx, SortValuesArgs &args);
  };

 public:
  static const int TASK_ID = OpCode::SORT_VALUES;

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

}  // namespace sorting
}  // namespace pandas
}  // namespace legate

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
namespace copy {

class DropNaTask : public PandasTask<DropNaTask> {
 public:
  struct DropNaTaskArgs {
    void sanity_check(void);

    using DropNaArg = std::pair<OutputColumn, Column<true>>;
    uint32_t keep_threshold;
    std::vector<int32_t> key_indices;
    std::vector<DropNaArg> pairs;
    FromFuture<int64_t> range_start;
    FromFuture<int64_t> range_step;
    friend void deserialize(Deserializer &ctx, DropNaTaskArgs &args);
  };

 public:
  static const int TASK_ID = OpCode::DROPNA;

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

}  // namespace copy
}  // namespace pandas
}  // namespace legate

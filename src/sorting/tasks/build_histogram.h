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

class BuildHistogramTask : public PandasTask<BuildHistogramTask> {
 public:
  struct BuildHistogramArgs {
    ~BuildHistogramArgs(void) { cleanup(); }
    void cleanup(void);
    void sanity_check(void);

    using Table = std::vector<Column<true>>;

    uint32_t num_pieces;
    bool put_null_first;
    std::vector<bool> ascending;
    Table samples;
    Table input;
    AccessorWO<Legion::Rect<1>, 2> hist_acc;
    Legion::Rect<2> hist_rect;

    friend void deserialize(Deserializer &ctx, BuildHistogramArgs &args);
  };

 public:
  static const int TASK_ID = OpCode::BUILD_HISTOGRAM;

 public:
  static void cpu_variant(const Legion::Task *task,
                          const std::vector<Legion::PhysicalRegion> &regions,
                          Legion::Context context,
                          Legion::Runtime *runtime);
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(const Legion::Task *task,
                          const std::vector<Legion::PhysicalRegion> &regions,
                          Legion::Context context,
                          Legion::Runtime *runtime);
#endif
};

}  // namespace sorting
}  // namespace pandas
}  // namespace legate

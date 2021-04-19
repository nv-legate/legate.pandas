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
#include "scalar/scalar.h"

namespace legate {
namespace pandas {
namespace copy {

class ScatterByMaskTask : public PandasTask<ScatterByMaskTask> {
 public:
  struct ScatterByMaskTaskArgs {
    void sanity_check(void);

    struct ScatterByMaskArg {
      OutputColumn output;
      Column<true> target;
      Column<true> input;
      Scalar scalar_input;
    };
    bool input_is_scalar;
    Column<true> mask;
    std::vector<ScatterByMaskArg> requests;
    friend void deserialize(Deserializer &ctx, ScatterByMaskTaskArgs &args);
  };

 public:
  static const int TASK_ID = OpCode::SCATTER_BY_MASK;

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

}  // namespace copy
}  // namespace pandas
}  // namespace legate

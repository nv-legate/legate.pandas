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
namespace io {
namespace csv {

class ToCSVTask : public PandasTask<ToCSVTask> {
 public:
  struct ToCSVArgs {
    uint32_t num_pieces;
    uint32_t chunksize;
    bool partition;
    bool header;
    std::string path;
    std::string sep;
    std::string na_rep;
    std::string line_terminator;
    std::vector<std::string> column_names;
    std::vector<Column<true>> columns;

    friend void deserialize(Deserializer &ctx, ToCSVArgs &args);
  };

 public:
  static const int TASK_ID = OpCode::TO_CSV;

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

}  // namespace csv
}  // namespace io
}  // namespace pandas
}  // namespace legate

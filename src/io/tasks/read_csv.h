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
namespace io {
namespace csv {

class ReadCSVTask : public PandasTask<ReadCSVTask> {
 public:
  struct ReadCSVArgs {
    using Columns = std::vector<OutputColumn>;

    std::vector<std::string> filenames;
    std::vector<CompressionType> compressions;
    std::string delimiter;
    int32_t skiprows;
    int32_t skipfooter;
    Maybe<int32_t> nrows;
    std::string quotechar;
    bool doublequote;
    bool skip_blank_lines;
    Maybe<std::vector<std::string>> true_values;
    Maybe<std::vector<std::string>> false_values;
    Maybe<std::vector<std::string>> na_values;
    Columns columns;
    std::vector<bool> is_string_column;
    std::vector<int32_t> date_columns;

    friend void deserialize(Deserializer &ctx, ReadCSVArgs &args);
  };

 public:
  static const int TASK_ID = OpCode::READ_CSV;

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

}  // namespace csv
}  // namespace io
}  // namespace pandas
}  // namespace legate

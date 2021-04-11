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

#include "pandas.h"
#include "deserializer.h"
#include "column/column.h"

namespace legate {
namespace pandas {
namespace partition {
namespace detail {

using InputTable  = std::vector<Column<true>>;
using OutputTable = std::vector<OutputColumn>;

struct LocalPartitionArgs {
  ~LocalPartitionArgs(void) { cleanup(); }
  void cleanup(void);
  void sanity_check(void);

  AccessorWO<Legion::Rect<1>, 2> hist_acc;
  Legion::Rect<2> hist_rect;
  uint32_t num_pieces;
  std::vector<int32_t> key_indices;
  InputTable input;
  OutputTable output;

  friend void deserialize(Deserializer &ctx, LocalPartitionArgs &args);
};

}  // namespace detail
}  // namespace partition
}  // namespace pandas
}  // namespace legate

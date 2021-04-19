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

#include "partitioning/local_partition_args.h"

namespace legate {
namespace pandas {
namespace partition {

using namespace Legion;

namespace detail {

void LocalPartitionArgs::sanity_check(void)
{
  for (auto &column : input) assert(input[0].shape() == column.shape());
}

void deserialize(Deserializer &ctx, LocalPartitionArgs &args)
{
  args.hist_rect = deserialize(ctx, args.hist_acc);

  deserialize(ctx, args.num_pieces);
  deserialize(ctx, args.key_indices);

  deserialize(ctx, args.input);
  deserialize(ctx, args.output);

#ifdef DEBUG_PANDAS
  args.sanity_check();
#endif
}

}  // namespace detail
}  // namespace partition
}  // namespace pandas
}  // namespace legate

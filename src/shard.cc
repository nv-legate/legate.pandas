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

#include "shard.h"

namespace legate {
namespace pandas {
namespace sharding {

using namespace Legion;

class TilingShardingFunctor : public ShardingFunctor {
 public:
  TilingShardingFunctor() : ShardingFunctor() {}

 public:
  virtual Legion::ShardID shard(const Legion::DomainPoint& point,
                                const Legion::Domain& launch_space,
                                const size_t total_shards);
};

PandasShardingFunctor::PandasShardingFunctor() : ShardingFunctor() {}

/*static*/ void PandasShardingFunctor::register_sharding_functors(Runtime* runtime, ShardingID base)
{
  sharding_functors[static_cast<int>(ShardingCode::SHARD_TILE)] = new TilingShardingFunctor();
  for (int idx = 0; idx < NUM_SHARD; ++idx)
    runtime->register_sharding_functor(base + idx, sharding_functors[idx]);
}

/*static*/ ShardingFunctor* PandasShardingFunctor::sharding_functors[NUM_SHARD];

ShardID TilingShardingFunctor::shard(const Legion::DomainPoint& point,
                                     const Legion::Domain& launch_space,
                                     const size_t total_shards)
{
  const size_t num_tasks       = launch_space.get_volume();
  const size_t tasks_per_shard = (num_tasks + total_shards - 1) / total_shards;
  ShardID shard_id             = static_cast<ShardingID>(point[0]) / tasks_per_shard;
  return shard_id;
}

}  // namespace sharding
}  // namespace pandas
}  // namespace legate

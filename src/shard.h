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

namespace legate {
namespace pandas {
namespace sharding {

class PandasShardingFunctor : public Legion::ShardingFunctor {
 public:
  PandasShardingFunctor();

 public:
  virtual Legion::ShardID shard(const Legion::DomainPoint& point,
                                const Legion::Domain& launch_space,
                                const size_t total_shards) = 0;

 public:
  static void register_sharding_functors(Legion::Runtime* runtime, Legion::ShardingID base);

 private:
  static Legion::ShardingFunctor* sharding_functors[NUM_SHARD];
};

}  // namespace sharding
}  // namespace pandas
}  // namespace legate

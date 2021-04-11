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
namespace projection {

struct PandasProjectionFunctor : public Legion::ProjectionFunctor {
 public:
  PandasProjectionFunctor(Legion::Runtime *rt);
  virtual Legion::LogicalRegion project(Legion::LogicalPartition upper_bound,
                                        const Legion::DomainPoint &point,
                                        const Legion::Domain &launch_domain);

 public:
  virtual Legion::DomainPoint project_point(const Legion::DomainPoint &point,
                                            const Legion::Domain &launch_domain) const = 0;

 public:
  static void register_projection_functors(Legion::Runtime *runtime, Legion::ProjectionID base);

 private:
  static Legion::ProjectionFunctor *proj_functors[NUM_PROJ];
};

}  // namespace projection
}  // namespace pandas
}  // namespace legate

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

#include "proj.h"

namespace legate {
namespace pandas {
namespace projection {

using namespace Legion;

template <int RADIX, int OFFSET>
class PandasProjectionFunctorRadix : public PandasProjectionFunctor {
 public:
  PandasProjectionFunctorRadix(Runtime *rt, ProjectionCode c) : PandasProjectionFunctor(rt), code(c)
  {
  }

 public:
  virtual bool is_functional(void) const { return true; }
  virtual bool is_exclusive(void) const { return true; }
  virtual unsigned get_depth(void) const { return 0; }

 public:
  virtual DomainPoint project_point(const DomainPoint &point, const Domain &launch_domain) const;

 public:
  const ProjectionCode code;
};

template <int RADIX, int OFFSET>
DomainPoint PandasProjectionFunctorRadix<RADIX, OFFSET>::project_point(
  const DomainPoint &p, const Domain &launch_domain) const
{
  const Point<1> point = p;
  Point<1> out         = point;
  out[0]               = point[0] * RADIX + OFFSET;
  return DomainPoint(out);
}

PandasProjectionFunctor::PandasProjectionFunctor(Legion::Runtime *rt) : ProjectionFunctor(rt) {}

LogicalRegion PandasProjectionFunctor::project(LogicalPartition upper_bound,
                                               const DomainPoint &point,
                                               const Domain &launch_domain)
{
  const DomainPoint dp = project_point(point, launch_domain);
  if (runtime->has_logical_subregion_by_color(upper_bound, dp))
    return runtime->get_logical_subregion_by_color(upper_bound, dp);
  else
    return LogicalRegion::NO_REGION;
}

/*static*/ ProjectionFunctor *PandasProjectionFunctor::proj_functors[NUM_PROJ];

#define REGISTER_RADIX_FUNCTOR(runtime, base, radix, offset)                              \
  {                                                                                       \
    ProjectionCode code = ProjectionCode::PROJ_RADIX_##radix##_##offset;                  \
    int idx             = static_cast<int>(code);                                         \
    proj_functors[idx]  = new PandasProjectionFunctorRadix<radix, offset>(runtime, code); \
  }

/*static*/ void PandasProjectionFunctor::register_projection_functors(Runtime *runtime,
                                                                      ProjectionID base)
{
  REGISTER_RADIX_FUNCTOR(runtime, base, 4, 0);
  REGISTER_RADIX_FUNCTOR(runtime, base, 4, 1);
  REGISTER_RADIX_FUNCTOR(runtime, base, 4, 2);
  REGISTER_RADIX_FUNCTOR(runtime, base, 4, 3);
  for (int idx = 0; idx < static_cast<int>(ProjectionCode::LAST_PROJ); ++idx)
    runtime->register_projection_functor(base + idx, proj_functors[idx]);
}

}  // namespace projection
}  // namespace pandas
}  // namespace legate

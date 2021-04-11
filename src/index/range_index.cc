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

#include <cmath>
#include <sstream>
#include "index/range_index.h"

namespace legate {
namespace pandas {
namespace index {

using namespace Legion;

void deserialize(Deserializer &ctx, RangeIndex &index)
{
  deserialize_from_future(ctx, index.start);
  deserialize_from_future(ctx, index.stop);
  deserialize_from_future(ctx, index.step);
}

std::string RangeIndex::tostring() const
{
  std::stringstream ss;
  ss << "RangeIndex(start=" << start << ", stop=" << stop << ", step=" << step << ")";
  return ss.str();
}

std::pair<int64_t, int64_t> RangeIndex::image(const Rect<1> &subrect) const
{
  if (bounds().intersection(subrect).empty()) return std::make_pair<int64_t, int64_t>(0, 0);

  auto lo = subrect.lo[0] * step + start;
  auto hi = (subrect.hi[0] + 1) * step + start;

  return std::make_pair<int64_t, int64_t>(lo, hi);
}

int64_t RangeIndex::offset(int64_t value) const
{
  return step >= 0 ? (value - start + (step - 1)) / step : (start - value + (-step - 1)) / -step;
}

Rect<1> RangeIndex::inverse_image(const std::pair<int64_t, int64_t> &subrange, bool exclusive) const
{
  return Rect<1>(offset(subrange.first), offset(subrange.second) - exclusive);
}

int64_t RangeIndex::volume() const { return offset(stop); }

Rect<1> RangeIndex::bounds() const { return Rect<1>(0, volume() - 1); }

}  // namespace index
}  // namespace pandas
}  // namespace legate

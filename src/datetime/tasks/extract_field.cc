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

#include "datetime/tasks/extract_field.h"
#include "column/column.h"
#include "util/allocator.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace datetime {

using namespace Legion;
using ColumnView = detail::Column;

namespace detail {

template <DatetimeFieldCode CODE>
struct Extractor;

template <>
struct Extractor<DatetimeFieldCode::YEAR> {
  // Copied from the cuDF code
  constexpr int16_t operator()(const int64_t& t) const
  {
    const int64_t units_per_day = 86400000000000L;

    const int z        = ((t >= 0 ? t : t - (units_per_day - 1)) / units_per_day) + 719468;
    const int era      = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned doe = static_cast<unsigned>(z - era * 146097);
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const int y        = static_cast<int>(yoe) + era * 400;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned mp  = (5 * doy + 2) / 153;
    const unsigned m   = mp + (mp < 10 ? 3 : -9);
    if (m <= 2)
      return y + 1;
    else
      return y;
  }
};

template <>
struct Extractor<DatetimeFieldCode::MONTH> {
  // Copied from the cuDF code
  constexpr int16_t operator()(const int64_t& t) const
  {
    const int64_t units_per_day = 86400000000000L;

    const int z        = ((t >= 0 ? t : t - (units_per_day - 1)) / units_per_day) + 719468;
    const int era      = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned doe = static_cast<unsigned>(z - era * 146097);
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned mp  = (5 * doy + 2) / 153;
    return mp + (mp < 10 ? 3 : -9);
  }
};

template <>
struct Extractor<DatetimeFieldCode::DAY> {
  // Copied from the cuDF code
  constexpr int16_t operator()(const int64_t& t) const
  {
    const int64_t units_per_day = 86400000000000L;

    const int z        = ((t >= 0 ? t : t - (units_per_day - 1)) / units_per_day) + 719468;
    const int era      = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned doe = static_cast<unsigned>(z - era * 146097);
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned mp  = (5 * doy + 2) / 153;
    return doy - (153 * mp + 2) / 5 + 1;
  }
};

template <>
struct Extractor<DatetimeFieldCode::HOUR> {
  // Copied from the cuDF code
  constexpr int16_t operator()(const int64_t& t) const
  {
    const int64_t units_per_day  = 86400000000000;
    const int64_t units_per_hour = 3600000000000;

    return t >= 0 ? ((t % units_per_day) / units_per_hour)
                  : ((units_per_day + (t % units_per_day)) / units_per_hour);
  }
};

template <>
struct Extractor<DatetimeFieldCode::MINUTE> {
  // Copied from the cuDF code
  constexpr int16_t operator()(const int64_t& t) const
  {
    const int64_t units_per_hour   = 3600000000000;
    const int64_t units_per_minute = 60000000000;

    return t >= 0 ? ((t % units_per_hour) / units_per_minute)
                  : ((units_per_hour + (t % units_per_hour)) / units_per_minute);
  }
};

template <>
struct Extractor<DatetimeFieldCode::SECOND> {
  // Copied from the cuDF code
  constexpr int16_t operator()(const int64_t& t) const
  {
    const int64_t units_per_minute = 60000000000;
    const int64_t units_per_second = 1000000000;

    return t >= 0 ? ((t % units_per_minute) / units_per_second)
                  : ((units_per_minute + (t % units_per_minute)) / units_per_second);
  }
};

template <>
struct Extractor<DatetimeFieldCode::WEEKDAY> {
  // Copied from the cuDF code
  constexpr int16_t operator()(const int64_t& t) const
  {
    const int64_t units_per_day = 86400000000000L;

    const int z        = ((t >= 0 ? t : t - (units_per_day - 1)) / units_per_day) + 719468;
    const int era      = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned doe = static_cast<unsigned>(z - era * 146097);
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned mp  = (5 * doy + 2) / 153;
    const unsigned d   = doy - (153 * mp + 2) / 5 + 1;
    int m              = mp + (mp < 10 ? 3 : -9);
    int y              = static_cast<int>(yoe) + era * 400;

    // apply Zeller's algorithm
    if (m <= 2) { y += 1; }

    if (m == 1) {
      m = 13;
      y -= 1;
    }

    if (m == 2) {
      m = 14;
      y -= 1;
    }

    const unsigned k = y % 100;
    const unsigned j = y / 100;
    const unsigned h = (d + 13 * (m + 1) / 5 + k + k / 4 + j / 4 + 5 * j) % 7;

    return (h - 2 + 7) % 7;  // pandas convention Monday = 0
  }
};

template <DatetimeFieldCode CODE>
ColumnView extract_field(const ColumnView& in, alloc::Allocator& allocator)
{
  auto size  = in.size();
  auto p_in  = in.column<int64_t>();
  auto p_out = allocator.allocate_elements<int16_t>(size);

  Extractor<CODE> extract{};

  if (in.nullable()) {
    auto in_b = in.bitmask();
    for (auto idx = 0; idx < size; ++idx) {
      if (!in_b.get(idx)) continue;
      p_out[idx] = extract(p_in[idx]);
    }
  } else
    for (auto idx = 0; idx < size; ++idx) p_out[idx] = extract(p_in[idx]);

  return ColumnView(TypeCode::INT16, p_out, size);
}

ColumnView extract_field(const ColumnView& in, DatetimeFieldCode code, alloc::Allocator& allocator)
{
  switch (code) {
    case DatetimeFieldCode::YEAR: {
      return extract_field<DatetimeFieldCode::YEAR>(in, allocator);
    }
    case DatetimeFieldCode::MONTH: {
      return extract_field<DatetimeFieldCode::MONTH>(in, allocator);
    }
    case DatetimeFieldCode::DAY: {
      return extract_field<DatetimeFieldCode::DAY>(in, allocator);
    }
    case DatetimeFieldCode::HOUR: {
      return extract_field<DatetimeFieldCode::HOUR>(in, allocator);
    }
    case DatetimeFieldCode::MINUTE: {
      return extract_field<DatetimeFieldCode::MINUTE>(in, allocator);
    }
    case DatetimeFieldCode::SECOND: {
      return extract_field<DatetimeFieldCode::SECOND>(in, allocator);
    }
    case DatetimeFieldCode::WEEKDAY: {
      return extract_field<DatetimeFieldCode::WEEKDAY>(in, allocator);
    }
  }
  assert(false);
  return ColumnView();
}

}  // namespace detail

/*static*/ void ExtractFieldTask::cpu_variant(const Task* task,
                                              const std::vector<PhysicalRegion>& regions,
                                              Context context,
                                              Runtime* runtime)
{
  Deserializer ctx{task, regions};

  DatetimeFieldCode code;
  OutputColumn out;
  Column<true> in;

  deserialize(ctx, code);
  deserialize(ctx, out);
  deserialize(ctx, in);

  if (in.empty()) {
    out.make_empty(true);
    return;
  }

  alloc::DeferredBufferAllocator allocator;
  auto result = detail::extract_field(in.view(), code, allocator);
  out.return_from_view(allocator, result);
}

static void __attribute__((constructor)) register_tasks(void)
{
  ExtractFieldTask::register_variants();
}

}  // namespace datetime
}  // namespace pandas
}  // namespace legate

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

#include "string/tasks/to_datetime.h"
#include "column/column.h"
#include "deserializer.h"

namespace legate {
namespace pandas {
namespace string {

using namespace Legion;

namespace detail {

// This string-to-timestamp coverter is copied and modified from the cuDF source code.
// (https://github.com/rapidsai/cudf/blob/branch-0.18/cpp/src/strings/convert/convert_datetime.cu)

/**
 * @brief  Units for timestamp conversion.
 * These are defined since there are more than what cudf supports.
 */
enum class timestamp_units {
  years,    ///< precision is years
  months,   ///< precision is months
  days,     ///< precision is days
  hours,    ///< precision is hours
  minutes,  ///< precision is minutes
  seconds,  ///< precision is seconds
  ms,       ///< precision is milliseconds
  us,       ///< precision is microseconds
  ns        ///< precision is nanoseconds
};

// used to index values in a timeparts array
enum timestamp_parse_component {
  TP_YEAR        = 0,
  TP_MONTH       = 1,
  TP_DAY         = 2,
  TP_DAY_OF_YEAR = 3,
  TP_HOUR        = 4,
  TP_MINUTE      = 5,
  TP_SECOND      = 6,
  TP_SUBSECOND   = 7,
  TP_TZ_MINUTES  = 8,
  TP_ARRAYSIZE   = 9
};

enum class format_char_type : int8_t {
  literal,   // literal char type passed through
  specifier  // timestamp format specifier
};

/**
 * @brief Represents a format specifier or literal from a timestamp format string.
 *
 * Created by the format_compiler when parsing a format string.
 */
struct alignas(4) format_item {
  format_char_type item_type;  // specifier or literal indicator
  char value;                  // specifier or literal value
  int8_t length;               // item length in bytes

  static format_item new_specifier(char format_char, int8_t length)
  {
    return format_item{format_char_type::specifier, format_char, length};
  }
  static format_item new_delimiter(char literal)
  {
    return format_item{format_char_type::literal, literal, 1};
  }
};

/**
 * @brief The format_compiler parses a timestamp format string into a vector of
 * format_items.
 *
 * The vector of format_items are used when parsing a string into timestamp
 * components and when formatting a string from timestamp components.
 */
struct format_compiler {
  std::string format;
  std::string template_string;
  std::vector<format_item> items;

  std::map<char, int8_t> specifier_lengths = {{'Y', 4},
                                              {'y', 2},
                                              {'m', 2},
                                              {'d', 2},
                                              {'H', 2},
                                              {'I', 2},
                                              {'M', 2},
                                              {'S', 2},
                                              {'f', 6},
                                              {'z', 5},
                                              {'Z', 3},
                                              {'p', 2},
                                              {'j', 3}};

  format_compiler(const std::string& fmt) : format(fmt)
  {
    const char* str = format.c_str();
    auto length     = format.length();
    while (length > 0) {
      char ch = *str++;
      length--;
      if (ch != '%') {
        items.push_back(format_item::new_delimiter(ch));
        template_string.append(1, ch);
        continue;
      }

      ch = *str++;
      length--;
      if (ch == '%')  // escaped % char
      {
        items.push_back(format_item::new_delimiter(ch));
        template_string.append(1, ch);
        continue;
      }
      if (ch >= '0' && ch <= '9') {
        specifier_lengths[*str] = static_cast<int8_t>(ch - '0');
        ch                      = *str++;
        length--;
      }

      int8_t spec_length = specifier_lengths[ch];
      items.push_back(format_item::new_specifier(ch, spec_length));
      template_string.append((size_t)spec_length, ch);
    }
  }

  format_item const* format_items() { return items.data(); }
  auto items_count() const { return items.size(); }
  int8_t subsecond_precision() const { return specifier_lengths.at('f'); }
};

// this parses date/time characters into a timestamp integer
struct parse_datetime {
  format_compiler compiler;
  format_item const* format_items;
  size_t items_count;
  timestamp_units units;
  int8_t subsecond_precision;

  parse_datetime(const std::string& fmt) : compiler(fmt)
  {
    format_items        = compiler.format_items();
    items_count         = compiler.items_count();
    subsecond_precision = compiler.subsecond_precision();
  }

  /**
   * @brief Return power of ten value given an exponent.
   *
   * @return `1x10^exponent` for `0 <= exponent <= 9`
   */
  constexpr int64_t power_of_ten(int32_t exponent)
  {
    constexpr int64_t powers_of_ten[] = {
      1L, 10L, 100L, 1000L, 10000L, 100000L, 1000000L, 10000000L, 100000000L, 1000000000L};
    return powers_of_ten[exponent];
  }

  //
  int32_t str2int(const char* str, int32_t bytes)
  {
    const char* ptr = str;
    int32_t value   = 0;
    for (int32_t idx = 0; idx < bytes; ++idx) {
      char chr = *ptr++;
      if (chr < '0' || chr > '9') break;
      value = (value * 10) + static_cast<int32_t>(chr - '0');
    }
    return value;
  }

  // Walk the format_items to read the datetime string.
  // Returns 0 if all ok.
  int parse_into_parts(const std::string& string, int32_t* timeparts)
  {
    auto ptr    = string.c_str();
    auto length = string.size();
    for (size_t idx = 0; idx < items_count; ++idx) {
      auto item = format_items[idx];
      if (item.value != 'f')
        item.length = static_cast<int8_t>(std::min(static_cast<size_t>(item.length), length));
      if (item.item_type == format_char_type::literal) {
        // static character we'll just skip;
        // consume item.length bytes from string
        ptr += item.length;
        length -= item.length;
        continue;
      }

      // special logic for each specifier
      switch (item.value) {
        case 'Y': timeparts[TP_YEAR] = str2int(ptr, item.length); break;
        case 'y': timeparts[TP_YEAR] = str2int(ptr, item.length) + 1900; break;
        case 'm': timeparts[TP_MONTH] = str2int(ptr, item.length); break;
        case 'd': timeparts[TP_DAY] = str2int(ptr, item.length); break;
        case 'j': timeparts[TP_DAY_OF_YEAR] = str2int(ptr, item.length); break;
        case 'H':
        case 'I': timeparts[TP_HOUR] = str2int(ptr, item.length); break;
        case 'M': timeparts[TP_MINUTE] = str2int(ptr, item.length); break;
        case 'S': timeparts[TP_SECOND] = str2int(ptr, item.length); break;
        case 'f': {
          int32_t const read_size = std::min(static_cast<size_t>(item.length), length);
          int64_t const fraction  = str2int(ptr, read_size) * power_of_ten(item.length - read_size);
          timeparts[TP_SUBSECOND] = static_cast<size_t>(fraction);
          break;
        }
        case 'p': {
          std::string am_pm(ptr, 2);
          auto hour = timeparts[TP_HOUR];
          if ((am_pm.compare("AM") == 0) || (am_pm.compare("am") == 0)) {
            if (hour == 12) hour = 0;
          } else if (hour < 12)
            hour += 12;
          timeparts[TP_HOUR] = hour;
          break;
        }
        case 'z': {
          int sign = *ptr == '-' ? 1 : -1;  // revert timezone back to UTC
          int hh   = str2int(ptr + 1, 2);
          int mm   = str2int(ptr + 3, 2);
          // ignoring the rest for now
          // item.length has how many chars we should read
          timeparts[TP_TZ_MINUTES] = sign * ((hh * 60) + mm);
          break;
        }
        case 'Z': break;  // skip
        default: return 3;
      }
      ptr += item.length;
      length -= item.length;
    }
    return 0;
  }

  int64_t timestamp_from_parts(int32_t const* timeparts, timestamp_units units)
  {
    auto year = timeparts[TP_YEAR];
    if (units == timestamp_units::years) return year - 1970;
    auto month = timeparts[TP_MONTH];
    if (units == timestamp_units::months)
      return ((year - 1970) * 12) + (month - 1);  // months are 1-12, need to 0-base it here
    auto day = timeparts[TP_DAY];
    // The months are shifted so that March is the starting month and February
    // (possible leap day in it) is the last month for the linear calculation
    year -= (month <= 2) ? 1 : 0;
    // date cycle repeats every 400 years (era)
    constexpr int32_t erasInDays  = 146097;
    constexpr int32_t erasInYears = (erasInDays / 365);
    auto era                      = (year >= 0 ? year : year - 399) / erasInYears;
    auto yoe                      = year - era * erasInYears;
    auto doy = month == 0 ? day : ((153 * (month + (month > 2 ? -3 : 9)) + 2) / 5 + day - 1);
    auto doe = (yoe * 365) + (yoe / 4) - (yoe / 100) + doy;
    int32_t days =
      (era * erasInDays) + doe - 719468;  // 719468 = days from 0000-00-00 to 1970-03-01
    if (units == timestamp_units::days) return days;

    auto tzadjust = timeparts[TP_TZ_MINUTES];  // in minutes
    auto hour     = timeparts[TP_HOUR];
    if (units == timestamp_units::hours) return (days * 24L) + hour + (tzadjust / 60);

    auto minute = timeparts[TP_MINUTE];
    if (units == timestamp_units::minutes)
      return static_cast<int64_t>(days * 24L * 60L) + (hour * 60L) + minute + tzadjust;

    auto second = timeparts[TP_SECOND];
    int64_t timestamp =
      (days * 24L * 3600L) + (hour * 3600L) + (minute * 60L) + second + (tzadjust * 60);
    if (units == timestamp_units::seconds) return timestamp;

    int64_t subsecond =
      timeparts[TP_SUBSECOND] * power_of_ten(9 - subsecond_precision);  // normalize to nanoseconds
    if (units == timestamp_units::ms) {
      timestamp *= 1000L;
      subsecond = subsecond / 1000000L;
    } else if (units == timestamp_units::us) {
      timestamp *= 1000000L;
      subsecond = subsecond / 1000L;
    } else if (units == timestamp_units::ns)
      timestamp *= 1000000000L;
    timestamp += subsecond;
    return timestamp;
  }

  int64_t operator()(const std::string& string)
  {
    int32_t timeparts[TP_ARRAYSIZE] = {1970, 1, 1};
    if (parse_into_parts(string, timeparts)) return 0;
    // FIXME: For now we hard-code the unit to nanoseconds
    return timestamp_from_parts(timeparts, timestamp_units::ns);
  }
};

}  // namespace detail

/*static*/ void ToDatetimeTask::cpu_variant(const Task* task,
                                            const std::vector<PhysicalRegion>& regions,
                                            Context context,
                                            Runtime* runtime)
{
  Deserializer ctx{task, regions};

  std::string format;
  deserialize(ctx, format);

  OutputColumn out;
  Column<true> in;
  deserialize(ctx, out);
  deserialize(ctx, in);

  const auto size = in.num_elements();
  if (size == 0) {
    out.make_empty();
    return;
  }

  out.allocate(size);
  auto out_ts = out.raw_column<int64_t>();

  detail::parse_datetime parser{format};
  auto in_offsets = in.child(0).raw_column_read<int32_t>();
  auto in_chars   = in.child(1).raw_column_read<int8_t>();

  if (in.nullable()) {
    auto in_b = in.read_bitmask();
    for (auto i = 0; i < size; ++i) {
      if (!in_b.get(i)) continue;
      std::string str{&in_chars[in_offsets[i]], &in_chars[in_offsets[i + 1]]};
      out_ts[i] = parser(str);
    }
  } else {
    for (auto i = 0; i < size; ++i) {
      std::string str{&in_chars[in_offsets[i]], &in_chars[in_offsets[i + 1]]};
      out_ts[i] = parser(str);
    }
  }
}

static void __attribute__((constructor)) register_tasks(void)
{
  ToDatetimeTask::register_variants();
}

}  // namespace string
}  // namespace pandas
}  // namespace legate

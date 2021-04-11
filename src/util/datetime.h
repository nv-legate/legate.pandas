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

// This is a datetime library implementation copied from libcudf and modified

#pragma once

inline long findFirstOccurrence(const char *data, long start_idx, long end_idx, char c)
{
  for (long i = start_idx; i <= end_idx; ++i) {
    if (data[i] == c) { return i; }
  }

  return -1;
}

template <typename T>
T convertStrToInteger(const char *data, long start, long end)
{
  T value = 0;

  long index = start;
  while (index <= end) {
    if (data[index] >= '0' && data[index] <= '9') {
      value *= 10;
      value += data[index] - '0';
    }
    ++index;
  }

  return value;
}

// User-defined literals to clarify numbers and units for time calculation
constexpr uint32_t operator"" _days(unsigned long long int days) { return days; }
constexpr uint32_t operator"" _erasInDays(unsigned long long int eras)
{
  return eras * 146097_days;  // multiply by days within an era (400 year span)
}
constexpr uint32_t operator"" _years(unsigned long long int years) { return years; }
constexpr uint32_t operator"" _erasInYears(unsigned long long int eras)
{
  return (eras * 1_erasInDays) / 365_days;
}

constexpr int64_t daysSinceBaseline(int64_t year, int64_t month, int64_t day)
{
  // More details of this formula are located in cuDF datetime_ops
  // In brief, the calculation is split over several components:
  //     era: a 400 year range, where the date cycle repeats exactly
  //     yoe: year within the 400 range of an era
  //     doy: day within the 364 range of a year
  //     doe: exact day within the whole era
  // The months are shifted so that March is the starting month and February
  // (possible leap day in it) is the last month for the linear calculation
  year -= (month <= 2) ? 1 : 0;

  const int64_t era = (year >= 0 ? year : year - 399_years) / 1_erasInYears;
  const int64_t yoe = year - era * 1_erasInYears;
  const int64_t doy = (153_days * (month + (month > 2 ? -3 : 9)) + 2) / 5 + day - 1;
  const int64_t doe = (yoe * 365_days) + (yoe / 4_years) - (yoe / 100_years) + doy;

  return (era * 1_erasInDays) + doe;
}

constexpr int64_t daysSinceEpoch(int64_t year, int64_t month, int64_t day)
{
  // Shift the start date to epoch to match unix time
  static_assert(daysSinceBaseline(1970, 1, 1) == 719468_days,
                "Baseline to epoch returns incorrect number of days");

  return daysSinceBaseline(year, month, day) - daysSinceBaseline(1970, 1, 1);
}

constexpr int64_t secondsSinceEpoch(
  int64_t year, int64_t month, int64_t day, int64_t hour, int64_t minute, int64_t second)
{
  // Leverage the function to find the days since epoch
  const auto days = daysSinceEpoch(year, month, day);

  // Return sum total seconds from each time portion
  return (days * 24 * 60 * 60) + (hour * 60 * 60) + (minute * 60) + second;
}

inline bool extractDate(const char *data,
                        long sIdx,
                        long eIdx,
                        bool dayfirst,
                        int64_t *year,
                        int64_t *month,
                        int64_t *day)
{
  char sep = '/';

  long sep_pos = findFirstOccurrence(data, sIdx, eIdx, sep);

  if (sep_pos == -1) {
    sep     = '-';
    sep_pos = findFirstOccurrence(data, sIdx, eIdx, sep);
  }

  if (sep_pos == -1) return false;

  //--- is year the first filed?
  if ((sep_pos - sIdx) == 4) {
    *year = convertStrToInteger<int64_t>(data, sIdx, (sep_pos - 1));

    // Month
    long s2 = sep_pos + 1;
    sep_pos = findFirstOccurrence(data, s2, eIdx, sep);

    if (sep_pos == -1) {
      //--- Data is just Year and Month - no day
      *month = convertStrToInteger<int64_t>(data, s2, eIdx);
      *day   = 1;

    } else {
      *month = convertStrToInteger<int64_t>(data, s2, (sep_pos - 1));
      *day   = convertStrToInteger<int64_t>(data, (sep_pos + 1), eIdx);
    }

  } else {
    //--- if the dayfirst flag is set, then restricts the format options
    if (dayfirst) {
      *day = convertStrToInteger<int64_t>(data, sIdx, (sep_pos - 1));

      long s2 = sep_pos + 1;
      sep_pos = findFirstOccurrence(data, s2, eIdx, sep);

      *month = convertStrToInteger<int64_t>(data, s2, (sep_pos - 1));
      *year  = convertStrToInteger<int64_t>(data, (sep_pos + 1), eIdx);

    } else {
      *month = convertStrToInteger<int64_t>(data, sIdx, (sep_pos - 1));

      long s2 = sep_pos + 1;
      sep_pos = findFirstOccurrence(data, s2, eIdx, sep);

      if (sep_pos == -1) {
        //--- Data is just Year and Month - no day
        *year = convertStrToInteger<int64_t>(data, s2, eIdx);
        *day  = 1;

      } else {
        *day  = convertStrToInteger<int64_t>(data, s2, (sep_pos - 1));
        *year = convertStrToInteger<int64_t>(data, (sep_pos + 1), eIdx);
      }
    }
  }

  return true;
}

inline void extractTime(const char *data,
                        long start,
                        long end,
                        int64_t *hour,
                        int64_t *minute,
                        int64_t *second,
                        int64_t *millisecond)
{
  constexpr char sep = ':';

  // Adjust for AM/PM and any whitespace before
  int64_t hour_adjust = 0;
  if (data[end] == 'M' || data[end] == 'm') {
    if (data[end - 1] == 'P' || data[end - 1] == 'p') { hour_adjust = 12; }
    end = end - 2;
    while (data[end] == ' ') { --end; }
  }

  // Find hour-minute separator
  const auto hm_sep = findFirstOccurrence(data, start, end, sep);
  *hour             = convertStrToInteger<int64_t>(data, start, hm_sep - 1) + hour_adjust;

  // Find minute-second separator (if present)
  const auto ms_sep = findFirstOccurrence(data, hm_sep + 1, end, sep);
  if (ms_sep == -1) {
    *minute      = convertStrToInteger<int64_t>(data, hm_sep + 1, end);
    *second      = 0;
    *millisecond = 0;
  } else {
    *minute = convertStrToInteger<int64_t>(data, hm_sep + 1, ms_sep - 1);

    // Find second-millisecond separator (if present)
    const auto sms_sep = findFirstOccurrence(data, ms_sep + 1, end, '.');
    if (sms_sep == -1) {
      *second      = convertStrToInteger<int64_t>(data, ms_sep + 1, end);
      *millisecond = 0;
    } else {
      *second      = convertStrToInteger<int64_t>(data, ms_sep + 1, sms_sep - 1);
      *millisecond = convertStrToInteger<int64_t>(data, sms_sep + 1, end);
    }
  }
}

inline int32_t parseDateFormat(const char *data, long start_idx, long end_idx, bool dayfirst)
{
  int64_t day, month, year;
  int32_t e = -1;

  bool status = extractDate(data, start_idx, end_idx, dayfirst, &year, &month, &day);

  if (status) e = daysSinceEpoch(year, month, day);

  return e;
}

inline int64_t parseDateTimeFormat(const char *data, long start, long end, bool dayfirst)
{
  int64_t day, month, year;
  int64_t hour, minute, second, millisecond = 0;
  int64_t answer = -1;

  // Find end of the date portion
  // TODO: Refactor all the date/time parsing to remove multiple passes over
  // each character because of find() then convert(); that can also avoid the
  // ugliness below.
  auto sep_pos = findFirstOccurrence(data, start, end, 'T');
  if (sep_pos == -1) {
    // Attempt to locate the position between date and time, ignore premature
    // space separators around the day/month/year portions
    int64_t count = 0;
    for (long i = start; i <= end; ++i) {
      if (count == 3 && data[i] == ' ') {
        sep_pos = i;
        break;
      } else if ((data[i] == '/' || data[i] == '-') || (count == 2 && data[i] != ' ')) {
        count++;
      }
    }
  }

  // There is only date if there's no separator, otherwise it's malformed
  if (sep_pos != -1) {
    if (extractDate(data, start, sep_pos - 1, dayfirst, &year, &month, &day)) {
      extractTime(data, sep_pos + 1, end, &hour, &minute, &second, &millisecond);
      answer = secondsSinceEpoch(year, month, day, hour, minute, second) * 1000 + millisecond;
    }
  } else {
    if (extractDate(data, start, end, dayfirst, &year, &month, &day)) {
      answer = secondsSinceEpoch(year, month, day, 0, 0, 0) * 1000;
    }
  }
  std::string year_str(&data[end - 3], &data[end + 1]);
  int64_t year_parsed = atol(year_str.c_str());
  assert(year == year_parsed);

  return answer;
}

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

#include <time.h>
#include "datetime/utilities.h"

namespace legate {
namespace pandas {
namespace datetime {
namespace detail {

std::string to_string(const int64_t &value)
{
  // FIXME: This is hacky and does not work for a format string other than the one used below.
  time_t epoch = value / 1000000000L + 28800;
  auto time    = localtime(&epoch);
  char date[128];
  strftime(date, sizeof(date), "%Y-%m-%d %H:%M:%S", time);
  return date;
}

}  // namespace detail
}  // namespace datetime
}  // namespace pandas
}  // namespace legate

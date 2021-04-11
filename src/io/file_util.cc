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
#include <iomanip>
#include <sstream>

#include "io/file_util.h"

namespace legate {
namespace pandas {
namespace io {

std::string get_partition_filename(std::string &&prefix,
                                   std::string &&delim,
                                   std::string &&suffix,
                                   uint32_t total_num_pieces,
                                   uint32_t task_id)
{
  std::stringstream ss;
  auto num_digits = static_cast<uint32_t>(std::log10(total_num_pieces)) + 1;
  ss << prefix << delim << "part" << std::setfill('0') << std::setw(num_digits) << task_id
     << suffix;

  return ss.str();
}

}  // namespace io
}  // namespace pandas
}  // namespace legate

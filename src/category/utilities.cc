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

#include "category/utilities.h"

namespace legate {
namespace pandas {
namespace category {

void to_dictionary(std::vector<std::string> &dictionary, const detail::Column &column)
{
#ifdef DEBUG_PANDAS
  assert(column.code() == TypeCode::STRING);
#endif
  auto dict_size    = column.size();
  auto dict_offsets = column.child(0).column<int32_t>();
  auto dict_chars   = column.child(1).column<int8_t>();

  dictionary.resize(dict_size);
  for (auto i = 0; i < dict_size; ++i) {
    auto lo       = dict_offsets[i];
    auto hi       = dict_offsets[i + 1];
    dictionary[i] = std::string{&dict_chars[lo], &dict_chars[hi]};
  }
}

}  // namespace category
}  // namespace pandas
}  // namespace legate

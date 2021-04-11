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

#include <sstream>
#include <type_traits>

namespace legate {
namespace pandas {
namespace string {
namespace detail {

void float_to_string(std::stringstream &ss, double value);

template <typename T>
struct Converter {
  template <typename _T = T, std::enable_if_t<std::is_integral<_T>::value> * = nullptr>
  std::string operator()(const _T &v) const
  {
    std::stringstream ss;
    ss << std::dec << v;
    return ss.str();
  }

  template <typename _T = T, std::enable_if_t<std::is_floating_point<_T>::value> * = nullptr>
  std::string operator()(const _T &v) const
  {
    std::stringstream ss;
    float_to_string(ss, v);
    return ss.str();
  }
};

template <>
struct Converter<int8_t> {
  std::string operator()(const int8_t &v) const
  {
    std::stringstream ss;
    ss << std::dec << static_cast<int32_t>(v);
    return ss.str();
  }
};

template <>
struct Converter<uint8_t> {
  std::string operator()(const uint8_t &v) const
  {
    std::stringstream ss;
    ss << std::dec << static_cast<uint32_t>(v);
    return ss.str();
  }
};

template <>
struct Converter<bool> {
  std::string operator()(const bool &v) const { return v ? "True" : "False"; }
};

template <>
struct Converter<std::string> {
  std::string operator()(const std::string &v) const { return v; }
};

}  // namespace detail
}  // namespace string
}  // namespace pandas
}  // namespace legate

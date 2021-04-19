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

class Scalar {
 public:
  Scalar() = default;
  Scalar(const Scalar &other) noexcept;
  Scalar(Scalar &&other) noexcept;
  ~Scalar();

 public:
  Scalar &operator=(const Scalar &other) noexcept;
  Scalar &operator=(Scalar &&other) noexcept;

 public:
  template <typename T>
  Scalar(bool valid, T val, TypeCode code = pandas_type_code_of<T>) : valid_(valid), code_(code)
  {
#ifdef DEBUG_PANDAS
    assert(code_ != TypeCode::INVALID);
#endif
    value<T>() = val;
  }

  Scalar(bool valid, const std::string &val) : valid_(valid), code_(TypeCode::STRING)
  {
    value_ = reinterpret_cast<uint64_t>(new std::string(val));
  }

  Scalar(TypeCode code) : valid_(false), code_(code) {}

 private:
  void copy_value(const Scalar &other);
  void destroy();

 public:
  size_t legion_buffer_size() const;
  void legion_serialize(void *buffer) const;
  void legion_deserialize(const void *buffer);

 public:
  bool valid() const { return valid_; }
  void set_valid(bool valid) { valid_ = valid; }
  auto code() const { return code_; }

 public:
  template <typename T>
  T &value()
  {
    return *ptr<T>();
  }

  template <typename T>
  const T &value() const
  {
    return *ptr<T>();
  }

  template <typename T>
  T *ptr()
  {
#ifdef DEBUG_PANDAS
    assert(pandas_type_code_of<T> == to_storage_type_code(code_));
#endif
    if (code_ != TypeCode::STRING)
      return reinterpret_cast<T *>(&value_);
    else
      return reinterpret_cast<T *>(value_);
  }

  template <typename T>
  const T *ptr() const
  {
#ifdef DEBUG_PANDAS
    assert(pandas_type_code_of<T> == to_storage_type_code(code_));
#endif
    if (code_ != TypeCode::STRING)
      return reinterpret_cast<const T *>(&value_);
    else
      return reinterpret_cast<const T *>(value_);
  }

  const void *raw_ptr()
  {
    if (!valid_)
      return nullptr;
    else if (code_ != TypeCode::STRING)
      return &value_;
    else
      return reinterpret_cast<const void *>(value_);
  }

 private:
  int32_t valid_{false};
  TypeCode code_{TypeCode::INVALID};
  uint64_t value_{0};
};

}  // namespace pandas
}  // namespace legate

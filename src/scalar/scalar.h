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
  Scalar()               = default;
  Scalar(const Scalar &) = default;
  Scalar(Scalar &&other) : valid_(other.valid_), code_(other.code_), value_(other.value_) {}
  void destroy()
  {
    if (code_ == TypeCode::STRING) delete ptr<std::string>();
  }

 public:
  constexpr Scalar &operator=(Scalar &&other)
  {
    valid_ = other.valid_;
    code_  = other.code_;
    value_ = other.value_;
    return *this;
  }

 public:
  template <typename T>
  Scalar(bool valid, T value, TypeCode code = pandas_type_code_of<T>) : valid_(valid), code_(code)
  {
    *reinterpret_cast<T *>(&value_) = value;
  }

  Scalar(bool valid, std::string *value) : valid_(valid), code_(TypeCode::STRING)
  {
    *reinterpret_cast<std::string **>(&value_) = value;
  }

  Scalar(TypeCode code) : valid_(false), code_(code) {}

 public:
  inline size_t legion_buffer_size(void) const
  {
    if (code_ == TypeCode::STRING)
      return sizeof(Scalar) + value<std::string>().size();
    else
      return sizeof(Scalar);
  }

  inline void legion_serialize(void *buffer) const
  {
    if (code_ == TypeCode::STRING) {
      const auto header_size = sizeof(Scalar) - sizeof(value_);
      memcpy(buffer, this, header_size);

      auto payload_offset                         = static_cast<int8_t *>(buffer) + header_size;
      const auto val                              = value<std::string>();
      *reinterpret_cast<size_t *>(payload_offset) = val.size();
      memcpy(payload_offset + sizeof(size_t), val.c_str(), val.size());
    } else
      memcpy(buffer, this, sizeof(Scalar));
  }

  inline void legion_deserialize(const void *buffer)
  {
    valid_ = static_cast<const int32_t *>(buffer)[0];
    code_  = static_cast<TypeCode>(static_cast<const int32_t *>(buffer)[1]);

    if (!valid_) return;

    if (code_ == TypeCode::STRING) {
      auto len_offset  = static_cast<const int8_t *>(buffer) + sizeof(valid_) + sizeof(code_);
      auto char_offset = len_offset + sizeof(size_t);
      auto len         = *reinterpret_cast<const size_t *>(len_offset);
      auto value       = new std::string(char_offset, char_offset + len);
      *reinterpret_cast<std::string **>(&value_) = value;
    } else
      value_ = static_cast<const uint64_t *>(buffer)[1];
  }

 public:
  bool valid() const { return valid_; }
  void set_valid(bool valid) { valid_ = valid; }
  auto code() const { return code_; }

 public:
  template <typename T>
  T &value()
  {
#ifdef DEBUG_PANDAS
    assert(pandas_type_code_of<T> == to_storage_type_code(code_));
#endif
    if (code_ != TypeCode::STRING)
      return *reinterpret_cast<T *>(&value_);
    else
      return *reinterpret_cast<T *>(value_);
  }

  template <typename T>
  const T &value() const
  {
#ifdef DEBUG_PANDAS
    assert(pandas_type_code_of<T> == to_storage_type_code(code_));
#endif
    if (code_ != TypeCode::STRING)
      return *reinterpret_cast<const T *>(&value_);
    else
      return *reinterpret_cast<T *>(value_);
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

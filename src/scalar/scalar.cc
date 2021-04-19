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

#include "scalar/scalar.h"

namespace legate {
namespace pandas {

Scalar::Scalar(const Scalar &other) noexcept : valid_(other.valid_), code_(other.code_)
{
  copy_value(other);
}

Scalar::Scalar(Scalar &&other) noexcept
  : valid_(other.valid_), code_(other.code_), value_(other.value_)
{
  other.value_ = 0;
}

Scalar::~Scalar() { destroy(); }

Scalar &Scalar::operator=(const Scalar &other) noexcept
{
  destroy();
  valid_ = other.valid_;
  code_  = other.code_;
  copy_value(other);
  return *this;
}

Scalar &Scalar::operator=(Scalar &&other) noexcept
{
  destroy();
  valid_       = other.valid_;
  code_        = other.code_;
  value_       = other.value_;
  other.value_ = 0;
  return *this;
}

void Scalar::copy_value(const Scalar &other)
{
  if (!valid_) {
    value_ = 0;
    return;
  }
  if (code_ == TypeCode::STRING)
    value_ = reinterpret_cast<uint64_t>(new std::string(other.value<std::string>()));
  else
    value_ = other.value_;
}

void Scalar::destroy()
{
  if (code_ == TypeCode::STRING) delete ptr<std::string>();
}

size_t Scalar::legion_buffer_size(void) const
{
  if (code_ == TypeCode::STRING)
    return sizeof(Scalar) + value<std::string>().size();
  else
    return sizeof(Scalar);
}

void Scalar::legion_serialize(void *buffer) const
{
  if (code_ == TypeCode::STRING) {
    const auto header_size = sizeof(Scalar) - sizeof(value_);
    memcpy(buffer, this, header_size);

    auto payload_offset                         = static_cast<int8_t *>(buffer) + header_size;
    const auto &val                             = value<std::string>();
    *reinterpret_cast<size_t *>(payload_offset) = val.size();
    memcpy(payload_offset + sizeof(size_t), val.c_str(), val.size());
  } else
    memcpy(buffer, this, sizeof(Scalar));
}

void Scalar::legion_deserialize(const void *buffer)
{
  valid_ = static_cast<const int32_t *>(buffer)[0];
  code_  = static_cast<TypeCode>(static_cast<const int32_t *>(buffer)[1]);

  if (!valid_) return;

  if (code_ == TypeCode::STRING) {
    auto len_offset  = static_cast<const int8_t *>(buffer) + sizeof(valid_) + sizeof(code_);
    auto char_offset = len_offset + sizeof(size_t);
    auto len         = *reinterpret_cast<const size_t *>(len_offset);
    auto value       = new std::string(char_offset, char_offset + len);
    value_           = reinterpret_cast<uint64_t>(value);
  } else
    value_ = static_cast<const uint64_t *>(buffer)[1];
}

}  // namespace pandas
}  // namespace legate

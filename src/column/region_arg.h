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
#include "util/accessor_util.h"
#include "util/type_dispatch.h"

namespace legate {
namespace pandas {

template <bool READ, int DIM = 1>
class RegionArg {
 public:
  RegionArg();
  RegionArg(const RegionArg &other) = default;
  RegionArg(TypeCode code, const Legion::PhysicalRegion &pr, Legion::FieldID fid);

 public:
  const Legion::Rect<DIM> &shape() const
  {
    assert(domain_.dense());
    return domain_.bounds;
  }
  size_t size() const
  {
    assert(has_rect);
    return domain_.volume();
  }
  size_t empty() const
  {
    assert(has_rect);
    return domain_.empty();
  }
  size_t elem_size() const;
  size_t bytes() const;
  inline bool valid() const { return code != TypeCode::INVALID; }
  inline bool is_meta() const { return code == TypeCode::STRING || code == TypeCode::CAT32; }
  void destroy(void);

 public:
  void set_rect(const Legion::Rect<DIM> &rect);

 public:
  template <typename T>
  const AccessorRO<T, DIM> &read_accessor(void) const;
  template <typename T>
  const AccessorWO<T, DIM> &write_accessor(void) const;

 public:
  template <typename T>
  T *raw_write() const;
  template <typename T>
  const T *raw_read() const;

 public:
  void *raw_untyped_write() const;
  const void *raw_untyped_read() const;

 public:
  TypeCode code;

 private:
  Legion::PhysicalRegion pr_;
  Legion::FieldID fid_;
  bool has_rect;
  void *accessor_;  // Pointer to an untyped accessor
  Legion::DomainT<DIM> domain_;
};

}  // namespace pandas
}  // namespace legate

#include "column/region_arg.inl"

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

class OutputRegionArg {
 public:
  OutputRegionArg();
  OutputRegionArg(const OutputRegionArg &other) = default;
  OutputRegionArg(TypeCode code, const Legion::OutputRegion &out, Legion::FieldID fid);

 public:
  void destroy();

 public:
  void allocate(size_t num_elements, size_t alignment = 16);
  template <typename VAL>
  void return_from_buffer(Legion::DeferredBuffer<VAL, 1> &buffer, size_t num_elements);
  void return_from_instance(Realm::RegionInstance instance, size_t num_elements, size_t elem_size);

 public:
  Legion::Rect<1> shape() const;

 public:
  template <typename T>
  T *ptr() const;

 public:
  inline bool valid() const { return fid_ != -1U; }
  inline bool is_meta() const { return code == TypeCode::STRING || code == TypeCode::CAT32; }

 public:
  void *untyped_ptr() const;

 public:
  inline Legion::Memory target_memory() const { return out_.target_memory(); }
  TypeCode code;

 private:
  Legion::OutputRegion out_;
  Legion::FieldID fid_;
  void *buffer_;
};

}  // namespace pandas
}  // namespace legate

#include "output_region_arg.inl"

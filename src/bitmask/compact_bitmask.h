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

#include "legate.h"
#include "bitmask/bitmask.h"

#ifdef LEGATE_USE_CUDA
#include <cudf/null_mask.hpp>
#endif

namespace legate {
namespace pandas {

class CompactBitmask {
 public:
  using AllocType              = uint8_t;
  static const size_t NUM_BITS = sizeof(AllocType) * 8;

 public:
  CompactBitmask(AllocType* bitmask, size_t num_elements);
  CompactBitmask(const AllocType* bitmask, size_t num_elements);
  CompactBitmask(size_t num_elements, alloc::Allocator& allocator);

#ifdef LEGATE_USE_CUDA
 public:
  CompactBitmask(cudf::bitmask_type* bitmask, size_t num_elements);
  CompactBitmask(const cudf::bitmask_type* bitmask, size_t num_elements);
#endif

 public:
  LEGATE_DEVICE_PREFIX
  inline void set(size_t idx, bool v = true) const
  {
    AllocType& mask = find_submask(idx);
    size_t off      = idx % NUM_BITS;
    mask            = (mask & ~((AllocType)!v << off)) | (((AllocType)v) << off);
  }
  LEGATE_DEVICE_PREFIX
  inline constexpr bool get(size_t idx) const
  {
    return (find_submask(idx) >> (idx % NUM_BITS)) & 1;
  }

 private:
  LEGATE_DEVICE_PREFIX
  inline constexpr AllocType& find_submask(size_t idx) const { return bitmask[idx / NUM_BITS]; }

#ifdef LEGATE_USE_CUDA
 public:
  void to_boolmask(const Bitmask& target, cudaStream_t stream) const;
#endif

 public:
  auto raw_ptr() const { return bitmask; }

 private:
  AllocType* bitmask;

 public:
  const size_t num_elements;
  const size_t size;
};

}  // namespace pandas
}  // namespace legate

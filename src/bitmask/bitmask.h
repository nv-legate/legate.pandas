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
#include "util/allocator.h"
#ifdef LEGATE_USE_CUDA
#include <cudf/null_mask.hpp>
#endif

namespace legate {
namespace pandas {

class Bitmask {
 public:
  using AllocType = uint8_t;

 public:
  Bitmask(AllocType *bitmask, size_t num_elements);
  Bitmask(const AllocType *bitmask, size_t num_elements);
  Bitmask(size_t num_elements,
          alloc::Allocator &allocator,
          bool init       = false,
          bool init_value = false);

 public:
  Bitmask(const Bitmask &other) = default;

 public:
  LEGATE_DEVICE_PREFIX
  inline void set(size_t idx, bool v = true) const { bitmask[idx] = static_cast<AllocType>(v); }
  LEGATE_DEVICE_PREFIX
  inline constexpr bool get(size_t idx) const { return static_cast<bool>(bitmask[idx]); }

 public:
  void set_all_valid(void);
  void clear(void);
  size_t count_set_bits(void) const;
  inline size_t count_unset_bits(void) const { return num_elements - count_set_bits(); }
  void copy(const Bitmask &target) const;
  friend void intersect_bitmasks(Bitmask &out, const Bitmask &in1, const Bitmask &in2);

#ifdef LEGATE_USE_CUDA
 public:
  void set_all_valid(cudaStream_t stream);
  void clear(cudaStream_t stream);
  size_t count_unset_bits(cudaStream_t stream) const;
  void copy(const Bitmask &target, cudaStream_t stream) const;
  friend void intersect_bitmasks(Bitmask &out,
                                 const Bitmask &in1,
                                 const Bitmask &in2,
                                 cudaStream_t stream);
#endif

 public:
  auto raw_ptr() const { return bitmask; }

 private:
  AllocType *bitmask;

 public:
  const size_t num_elements;
};

}  // namespace pandas
}  // namespace legate

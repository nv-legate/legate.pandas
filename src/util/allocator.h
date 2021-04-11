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

#include <unordered_map>
#include <unordered_set>

#include "pandas.h"

namespace legate {
namespace pandas {
namespace alloc {

struct Allocator {
  template <typename T>
  T* allocate_elements(size_t num_elements)
  {
    return static_cast<T*>(allocate(sizeof(T) * num_elements));
  }
  virtual void* allocate(size_t bytes) = 0;
  virtual void deallocate(void* p)     = 0;
};

class DeferredBufferAllocator : public Allocator {
 public:
  using Buffer = Legion::DeferredBuffer<int8_t, 1>;

 public:
  DeferredBufferAllocator() = default;
  DeferredBufferAllocator(Legion::Memory::Kind kind);
  virtual ~DeferredBufferAllocator();

 public:
  virtual void* allocate(size_t bytes) override;
  virtual void deallocate(void* p) override;
  bool is_popped(const void* p) const;
  Buffer pop_allocation(const void* p);

 private:
  Legion::Memory::Kind target_kind{Legion::Memory::Kind::SYSTEM_MEM};
  std::unordered_map<const void*, Buffer> buffers{};
#ifdef DEBUG_PANDAS
  std::unordered_set<const void*> removed;
#endif
};

}  // namespace alloc
}  // namespace pandas
}  // namespace legate

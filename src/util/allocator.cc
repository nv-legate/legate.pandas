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

#include "util/allocator.h"

namespace legate {
namespace pandas {
namespace alloc {

using namespace Legion;

DeferredBufferAllocator::DeferredBufferAllocator(Memory::Kind kind) : target_kind(kind) {}

DeferredBufferAllocator::~DeferredBufferAllocator()
{
  for (auto& pair : buffers) { pair.second.destroy(); }
  buffers.clear();
}

void* DeferredBufferAllocator::allocate(size_t bytes)
{
  if (bytes == 0) return nullptr;

  // Use 16-byte alignment
  bytes = (bytes + 15) / 16 * 16;
  Rect<1> bounds(Point<1>(0), Point<1>(bytes - 1));

  Buffer buffer(target_kind, Domain(bounds));
  void* ptr = buffer.ptr(0);
#ifdef DEBUG_PANDAS
  assert(buffers.find(ptr) == buffers.end());
#endif
  buffers[ptr] = buffer;
  return ptr;
}

void DeferredBufferAllocator::deallocate(void* p)
{
  Buffer buffer;
  auto finder = buffers.find(p);
#ifdef DEBUG_PANDAS
  assert(finder != buffers.end() || removed.find(p) != removed.end());
#endif
  if (finder == buffers.end()) return;
  buffer = finder->second;
  buffers.erase(finder);
  buffer.destroy();
}

bool DeferredBufferAllocator::is_popped(const void* p) const
{
  return buffers.find(p) == buffers.end();
}

DeferredBufferAllocator::Buffer DeferredBufferAllocator::pop_allocation(const void* p)
{
  auto finder = buffers.find(p);
#ifdef DEBUG_PANDAS
  assert(finder != buffers.end());
  removed.insert(finder->first);
#endif
  auto result = finder->second;
  buffers.erase(finder);
  return result;
}

}  // namespace alloc
}  // namespace pandas
}  // namespace legate

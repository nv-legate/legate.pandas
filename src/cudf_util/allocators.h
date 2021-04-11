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

#include "bitmask/compact_bitmask.h"
#include "column/column.h"
#include "cudf_util/bitmask.h"
#include "util/allocator.h"

#include <mutex>
#include <unordered_set>
#include <unordered_map>
#include <utility>

#include <rmm/mr/device/device_memory_resource.hpp>

namespace legate {
namespace pandas {

struct DelegateAllocator : public rmm::mr::device_memory_resource {
  virtual void* delegate_allocation(std::size_t bytes, rmm::cuda_stream_view stream) = 0;
};

struct SingletonAllocator : public DelegateAllocator {
  SingletonAllocator() = default;

  SingletonAllocator(const std::pair<void*, size_t>& alloc) : idx(0), alloc(alloc) {}

  SingletonAllocator(void* ptr, size_t size) : idx(0), alloc(std::make_pair(ptr, size)) {}

  virtual bool supports_streams() const noexcept { return false; }

  virtual bool supports_get_mem_info() const noexcept { return false; }

  virtual void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream)
  {
    assert(idx++ == 0);
    return alloc.first;
  }

  virtual void do_deallocate(void* p, std::size_t bytes, rmm::cuda_stream_view stream)
  {
    // do nothing as we don't need to deallocate
  }

  virtual std::pair<std::size_t, std::size_t> do_get_mem_info(rmm::cuda_stream_view stream) const
  {
    return std::make_pair(0, 0);
  }

  virtual void* delegate_allocation(std::size_t bytes, rmm::cuda_stream_view stream)
  {
    return do_allocate(bytes, stream);
  }

  unsigned idx;
  std::pair<void*, size_t> alloc;
};

struct DeferredBufferAllocator : public DelegateAllocator, alloc::DeferredBufferAllocator {
  using Buffer = Legion::DeferredBuffer<int8_t, 1>;

  DeferredBufferAllocator() : alloc::DeferredBufferAllocator(Legion::Memory::GPU_FB_MEM) {}
  virtual ~DeferredBufferAllocator() {}

  virtual bool supports_streams() const noexcept { return true; }

  virtual bool supports_get_mem_info() const noexcept { return false; }

  virtual void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream)
  {
    auto result = alloc::DeferredBufferAllocator::allocate(bytes);
    if (nullptr == result) return result;
#ifdef DEBUG_PANDAS
    assert(stream_map.find(result) == stream_map.end());
#endif
    stream_map[result] = stream;
    return result;
  }

  virtual void do_deallocate(void* p, std::size_t bytes, rmm::cuda_stream_view stream)
  {
    if (!alloc::DeferredBufferAllocator::is_popped(p)) {
#ifdef DEBUG_PANDAS
      assert(stream_map.find(p) != stream_map.end());
#endif
      // TODO: We really need deferred allocation and deallocation for CUDA kernels
      //       to avoid blocking here.
      stream_map[p].synchronize();
      stream_map.erase(p);
      alloc::DeferredBufferAllocator::deallocate(p);
    }
  }

  virtual std::pair<std::size_t, std::size_t> do_get_mem_info(rmm::cuda_stream_view stream) const
  {
    return std::make_pair(0, 0);
  }

  virtual void* delegate_allocation(std::size_t bytes, rmm::cuda_stream_view stream)
  {
    return do_allocate(bytes, stream);
  }

  std::unordered_map<void*, rmm::cuda_stream_view> stream_map;
};

struct ProxyAllocator : public DelegateAllocator {
 public:
  static ProxyAllocator* kind;

  using Buffer = Legion::DeferredBuffer<int8_t, 1>;
  using Func   = std::function<void(void*, size_t)>;

  ProxyAllocator(Func f) : buffer(nullptr), size(0), postamble(f) {}

  ProxyAllocator(Bitmask bitmask, cudaStream_t stream)
    : buffer(nullptr), size(0), postamble([bitmask, stream](void* buffer, size_t size) {
        util::to_boolmask(bitmask.raw_ptr(),
                          static_cast<Bitmask::AllocType*>(buffer),
                          bitmask.num_elements,
                          stream);
      })
  {
  }

  ~ProxyAllocator()
  {
    if (nullptr != buffer) postamble(buffer, size);
  }

  virtual bool supports_streams() const noexcept { return true; }

  virtual bool supports_get_mem_info() const noexcept { return false; }

  virtual void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream)
  {
    using namespace Legion;
    Rect<1> bounds(Point<1>(0), Point<1>(bytes - 1));
    Buffer buf(Memory::GPU_FB_MEM, Domain(bounds));
    buffer = buf.ptr(0);
    size   = bytes;
    return buffer;
  }

  virtual void do_deallocate(void* p, std::size_t bytes, rmm::cuda_stream_view stream)
  {
    // Do nothing as deferred buffers will be collected
    // automatically when the current executing task finishes
  }

  virtual std::pair<std::size_t, std::size_t> do_get_mem_info(rmm::cuda_stream_view stream) const
  {
    return std::make_pair(0, 0);
  }

  virtual void* delegate_allocation(std::size_t bytes, rmm::cuda_stream_view stream)
  {
    return do_allocate(bytes, stream);
  }

 private:
  ProxyAllocator() = default;

  void* buffer;
  size_t size;
  Func postamble;
};

struct CompositeAllocator : public rmm::mr::device_memory_resource {
  using child = std::unique_ptr<DelegateAllocator>;

  CompositeAllocator() = default;

  virtual bool supports_streams() const noexcept { return false; }

  virtual bool supports_get_mem_info() const noexcept { return false; }

  virtual void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream)
  {
    assert(idx < children.size());
    return children[idx++]->delegate_allocation(bytes, stream);
  }

  virtual void do_deallocate(void* p, std::size_t bytes, rmm::cuda_stream_view stream)
  {
    // Do nothing
  }

  virtual std::pair<std::size_t, std::size_t> do_get_mem_info(rmm::cuda_stream_view stream) const
  {
    return std::make_pair(0, 0);
  }

  template <typename Allocator, typename... Args>
  void add(Args... args)
  {
    children.push_back(std::make_unique<Allocator>(args...));
  }

  unsigned idx{0};
  std::vector<child> children{};
};

}  // namespace pandas
}  // namespace legate

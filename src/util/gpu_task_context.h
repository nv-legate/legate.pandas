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

#include "cudf_util/allocators.h"

namespace legate {
namespace pandas {

// This helper class is to make sure that each GPU task uses its own allocator
// for temporary allocations from libcudf during its execution. This class also
// creates a fresh stream to be used for kernels.

class GPUTaskContext {
 public:
  GPUTaskContext() : allocator_(new DeferredBufferAllocator())
  {
    cudaStreamCreate(&stream_);
    rmm::mr::set_current_device_resource(allocator_);
  }
  ~GPUTaskContext()
  {
    rmm::mr::set_current_device_resource(nullptr);
    delete allocator_;
    cudaStreamDestroy(stream_);
  }

  constexpr cudaStream_t stream() const { return stream_; }

 private:
  DeferredBufferAllocator *allocator_{nullptr};
  cudaStream_t stream_{0};
};

}  // namespace pandas
}  // namespace legate

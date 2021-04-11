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

#include "bitmask/bitmask.h"
#include "cudf_util/detail.h"
#include "util/cuda_helper.h"

#include <cudf/reduction.hpp>
#include <cudf/column/column_view.hpp>

namespace legate {
namespace pandas {

void Bitmask::set_all_valid(cudaStream_t stream)
{
  cudaMemsetAsync(bitmask, 0x01, num_elements, stream);
}

void Bitmask::clear(cudaStream_t stream) { cudaMemsetAsync(bitmask, 0x00, num_elements, stream); }

size_t Bitmask::count_unset_bits(cudaStream_t stream) const
{
  cudf::column_view boolmask{
    cudf::data_type{cudf::type_id::UINT8}, static_cast<cudf::size_type>(num_elements), bitmask};
  auto type_id                        = cudf::data_type{cudf::type_to_id<int32_t>()};
  rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource();
  auto out = cudf::detail::reduce(boolmask, cudf::make_sum_aggregation(), type_id, stream, mr);
  auto null_count = static_cast<cudf::scalar_type_t<int32_t> *>(out.get())->value(stream);
  assert(num_elements >= null_count);
  return num_elements - null_count;
}

void Bitmask::copy(const Bitmask &target, cudaStream_t stream) const
{
  cudaMemcpyAsync(target.bitmask, bitmask, num_elements, cudaMemcpyDeviceToDevice, stream);
}

static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  intersect_inplace(Bitmask out, Bitmask in)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= out.num_elements) return;
  out.set(idx, out.get(idx) && in.get(idx));
}

static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  intersect(Bitmask out, Bitmask in1, Bitmask in2)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= out.num_elements) return;
  out.set(idx, in1.get(idx) && in2.get(idx));
}

void intersect_bitmasks(Bitmask &out, const Bitmask &in1, const Bitmask &in2, cudaStream_t stream)
{
  const size_t blocks = (out.num_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  if (in1.bitmask == out.bitmask)
    intersect_inplace<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(out, in2);
  else
    intersect<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(out, in1, in2);
}

}  // namespace pandas
}  // namespace legate

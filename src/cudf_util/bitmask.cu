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

#include "cudf_util/bitmask.h"
#include "cudf_util/allocators.h"

#include "bitmask/bitmask.h"

#include <cudf/column/column_view.hpp>
#include <cudf/detail/transform.hpp>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

namespace legate {
namespace pandas {
namespace util {

void to_bitmask(Bitmask::AllocType* bitmask,
                const Bitmask::AllocType* raw_boolmask,
                size_t num_bits,
                cudaStream_t stream)
{
  cudf::column_view boolmask{
    cudf::data_type{cudf::type_id::BOOL8}, static_cast<cudf::size_type>(num_bits), raw_boolmask};

  SingletonAllocator mr{bitmask, num_bits};
  cudf::detail::bools_to_mask(boolmask, stream, &mr);
}

void to_boolmask(Bitmask::AllocType* boolmask,
                 const Bitmask::AllocType* raw_bitmask,
                 size_t num_bits,
                 cudaStream_t stream)
{
  CompactBitmask bitmask(raw_bitmask, num_bits);

  thrust::for_each(thrust::cuda::par.on(stream),
                   thrust::counting_iterator<size_t>(0),
                   thrust::counting_iterator<size_t>(num_bits),
                   [=] LEGATE_DEVICE_PREFIX(size_t i) { boolmask[i] = bitmask.get(i); });
}

}  // namespace util
}  // namespace pandas
}  // namespace legate

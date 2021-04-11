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

#include "pandas.h"

#include "category/encode.h"
#include "copy/materialize.cuh"

#include <cudf/table/table_view.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/sorting.hpp>
#include <cudf/join.hpp>

#include <rmm/exec_policy.hpp>

namespace legate {
namespace pandas {
namespace category {
namespace detail {

template <typename Index>
std::unique_ptr<cudf::column> iota(cudf::size_type size, cudaStream_t stream)
{
  auto out = cudf::make_numeric_column(
    cudf::data_type(cudf::type_to_id<Index>()), size, cudf::mask_state::UNALLOCATED, stream);

  cudf::mutable_column_view m_out = *out;

  auto start = thrust::make_counting_iterator<Index>(0);
  auto stop  = thrust::make_counting_iterator<Index>(static_cast<Index>(size));

  thrust::copy(rmm::exec_policy(stream), start, stop, m_out.begin<Index>());

  return std::move(out);
}

std::unique_ptr<cudf::column> encode(const cudf::column_view &input,
                                     const cudf::column_view &dictionary,
                                     cudaStream_t stream,
                                     rmm::mr::device_memory_resource *mr)
{
  auto temp_mr = rmm::mr::get_current_device_resource();

  cudf::hash_join joiner(cudf::table_view({dictionary}), cudf::null_equality::EQUAL, stream);
  auto indexers = joiner.left_join(cudf::table_view({input}), cudf::null_equality::EQUAL, stream);

  auto order          = iota<int32_t>(input.size(), stream);
  auto gathered_order = cudf::detail::gather(cudf::table_view({order->view()}),
                                             indexers.first->begin(),
                                             indexers.first->end(),
                                             cudf::out_of_bounds_policy::DONT_CHECK,
                                             stream);

  auto codes          = iota<uint32_t>(dictionary.size(), stream);
  auto gathered_codes = cudf::detail::gather(cudf::table_view({codes->view()}),
                                             indexers.second->begin(),
                                             indexers.second->end(),
                                             cudf::out_of_bounds_policy::NULLIFY,
                                             stream);

  auto sorted =
    cudf::detail::sort_by_key(cudf::table_view({gathered_order->view(), gathered_codes->view()}),
                              cudf::table_view({gathered_order->view()}),
                              {},
                              {},
                              stream,
                              mr)
      ->release();

  return std::move(sorted[1]);
}

}  // namespace detail
}  // namespace category
}  // namespace pandas
}  // namespace legate

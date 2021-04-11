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

#include "copy/materialize.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_device_view.cuh>

#include <rmm/exec_policy.hpp>

namespace legate {
namespace pandas {
namespace copy {

using namespace Legion;

namespace detail {
struct convert_fn_t {
  convert_fn_t(int64_t start, int64_t step) : start_(start), step_(step) {}

  constexpr int64_t operator()(int64_t idx) const { return start_ + idx * step_; }
  int64_t start_{0};
  int64_t step_{1};
};

}  // namespace detail

std::unique_ptr<cudf::column> materialize(const Rect<1> &rect,
                                          int64_t start,
                                          int64_t step,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource *mr)
{
  auto size = rect.volume();
  auto out  = cudf::make_numeric_column(cudf::data_type(cudf::type_id::INT64),
                                       static_cast<cudf::size_type>(size),
                                       cudf::mask_state::UNALLOCATED,
                                       stream,
                                       mr);

  cudf::mutable_column_view m_out = *out;

  detail::convert_fn_t fn{start, step};
  auto start_it =
    thrust::make_transform_iterator(thrust::make_counting_iterator<int64_t>(rect.lo[0]), fn);
  auto stop_it =
    thrust::make_transform_iterator(thrust::make_counting_iterator<int64_t>(rect.hi[0] + 1), fn);

  thrust::copy(rmm::exec_policy(stream), start_it, stop_it, m_out.begin<int64_t>());

  return std::move(out);
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

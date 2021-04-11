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

#include "index/search.cuh"
#include "cudf_util/detail.h"
#include "util/type_dispatch.h"

#include <cudf/column/column_view.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/binaryop.hpp>

#include <rmm/exec_policy.hpp>

namespace legate {
namespace pandas {
namespace index {

namespace detail {
namespace gpu {

template <typename T>
struct search_fn {
  search_fn(cudf::column_device_view in, T to_find, int64_t default_idx)
    : in_(in), to_find_(to_find), default_idx_(default_idx)
  {
  }

  __device__ int64_t operator()(int64_t idx) const
  {
    if (in_.element<T>(static_cast<cudf::size_type>(idx)) == to_find_)
      return idx;
    else
      return default_idx_;
  }

  cudf::column_device_view in_;
  T to_find_;
  int64_t default_idx_;
};

struct Search {
  template <TypeCode CODE, std::enable_if_t<is_primitive_type<CODE>::value> * = nullptr>
  Scalar operator()(cudf::column_view in, const Scalar &to_find, bool forward, cudaStream_t stream)
  {
    using VAL = pandas_type_of<CODE>;

    int64_t size = static_cast<int64_t>(in.size());
    if (size == 0) return Scalar(TypeCode::INT64);

    auto d_in = cudf::column_device_view::create(in, stream);
    int64_t default_idx{forward ? size : -1};
    search_fn<VAL> fn(*d_in, to_find.value<VAL>(), default_idx);

    cudf::data_type type_id{cudf::type_to_id<int64_t>()};
    auto indices =
      cudf::make_fixed_width_column(type_id, size, cudf::mask_state::UNALLOCATED, stream);
    auto indices_view = indices->mutable_view();
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<coord_t>(0),
                      thrust::make_counting_iterator<coord_t>(size),
                      indices_view.begin<int64_t>(),
                      fn);

    auto agg       = forward ? cudf::make_min_aggregation() : cudf::make_max_aggregation();
    auto result_sc = cudf::detail::reduce(indices->view(), agg, type_id, stream);
    auto result    = static_cast<cudf::scalar_type_t<int64_t> *>(result_sc.get())->value(stream);
    if (result == default_idx)
      return Scalar(TypeCode::INT64);
    else
      return Scalar(true, result);
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::STRING> * = nullptr>
  Scalar operator()(cudf::column_view in, const Scalar &to_find, bool forward, cudaStream_t stream)
  {
    assert(false);
    return Scalar(TypeCode::INT64);
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::CAT32> * = nullptr>
  Scalar operator()(cudf::column_view in, const Scalar &to_find, bool forward, cudaStream_t stream)
  {
    assert(false);
    return Scalar(TypeCode::INT64);
  }
};

}  // namespace gpu
}  // namespace detail

Scalar search(cudf::column_view in, const Scalar &to_find, bool forward, cudaStream_t stream)
{
  if (!to_find.valid()) return Scalar(TypeCode::INT64);
  return type_dispatch(to_find.code(), detail::gpu::Search{}, in, to_find, forward, stream);
}

}  // namespace index
}  // namespace pandas
}  // namespace legate

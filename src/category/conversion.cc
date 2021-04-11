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

#include "category/conversion.h"
#include "cudf_util/scalar.h"

#include <cudf/types.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/replace.hpp>

namespace legate {
namespace pandas {
namespace category {

std::unique_ptr<cudf::column> to_string_column(cudf::dictionary_column_view column,
                                               cudaStream_t stream,
                                               rmm::mr::device_memory_resource* mr)
{
  auto codes         = cudf::column_view(column.indices().type(),
                                 column.size(),
                                 column.indices().head(),
                                 column.null_mask(),
                                 column.null_count());
  auto codes_int32   = cudf::detail::cast(codes, cudf::data_type(cudf::type_id::INT32), stream);
  int32_t fill_value = -1;
  auto gather_map    = cudf::detail::replace_nulls(
    codes_int32->view(), *to_cudf_scalar<TypeCode::INT32>(&fill_value, stream), stream);
  auto gather_result = cudf::detail::gather(cudf::table_view{{column.keys()}},
                                            gather_map->view(),
                                            cudf::out_of_bounds_policy::NULLIFY,
                                            cudf::detail::negative_index_policy::NOT_ALLOWED,
                                            stream,
                                            mr)
                         ->release();
  return std::move(gather_result[0]);
}

}  // namespace category
}  // namespace pandas
}  // namespace legate

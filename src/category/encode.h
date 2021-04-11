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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

namespace legate {
namespace pandas {
namespace category {
namespace detail {

std::unique_ptr<cudf::column> encode(const cudf::column_view &input,
                                     const cudf::column_view &dictionary,
                                     cudaStream_t stream,
                                     rmm::mr::device_memory_resource *mr);

}  // namespace detail
}  // namespace category
}  // namespace pandas
}  // namespace legate

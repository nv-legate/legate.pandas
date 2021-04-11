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

#include <cudf/column/column.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>

namespace legate {
namespace pandas {
namespace category {

std::unique_ptr<cudf::column> to_string_column(
  cudf::dictionary_column_view column,
  cudaStream_t stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace category
}  // namespace pandas
}  // namespace legate

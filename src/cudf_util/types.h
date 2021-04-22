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

#include <memory>

#include "pandas.h"

namespace cudf {

class aggregation;
enum class binary_operator;
enum class type_id;
enum class duplicate_keep_option;

namespace io {

enum class compression_type;

}  // namespace io
}  // namespace cudf

namespace legate {
namespace pandas {

cudf::type_id to_cudf_type_id(TypeCode code);

std::unique_ptr<cudf::aggregation> to_cudf_agg(AggregationCode code);

cudf::binary_operator to_cudf_binary_op(AggregationCode code);

cudf::io::compression_type to_cudf_compression(CompressionType compression);

cudf::duplicate_keep_option to_cudf_keep_option(KeepMethod method);

}  // namespace pandas
}  // namespace legate

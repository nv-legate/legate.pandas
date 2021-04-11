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

#include "cudf_util/types.h"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/io/types.hpp>

namespace legate {
namespace pandas {

cudf::type_id to_cudf_type_id(TypeCode code)
{
  switch (code) {
    case TypeCode::BOOL: {
      return cudf::type_id::BOOL8;
    }
    case TypeCode::INT8: {
      return cudf::type_id::INT8;
    }
    case TypeCode::INT16: {
      return cudf::type_id::INT16;
    }
    case TypeCode::INT32: {
      return cudf::type_id::INT32;
    }
    case TypeCode::INT64: {
      return cudf::type_id::INT64;
    }
    case TypeCode::UINT8: {
      return cudf::type_id::UINT8;
    }
    case TypeCode::UINT16: {
      return cudf::type_id::UINT16;
    }
    case TypeCode::UINT32: {
      return cudf::type_id::UINT32;
    }
    case TypeCode::UINT64: {
      return cudf::type_id::UINT64;
    }
    case TypeCode::FLOAT: {
      return cudf::type_id::FLOAT32;
    }
    case TypeCode::DOUBLE: {
      return cudf::type_id::FLOAT64;
    }
    case TypeCode::TS_NS: {
      return cudf::type_id::TIMESTAMP_NANOSECONDS;
    }
    case TypeCode::STRING: {
      return cudf::type_id::STRING;
    }
    case TypeCode::CAT32: {
      return cudf::type_id::DICTIONARY32;
    }
  }
  assert(false);
  return cudf::type_id::EMPTY;
}

std::unique_ptr<cudf::aggregation> to_cudf_agg(AggregationCode code)
{
  switch (code) {
    case AggregationCode::SUM: {
      return cudf::make_sum_aggregation();
    }
    case AggregationCode::MIN: {
      return cudf::make_min_aggregation();
    }
    case AggregationCode::MAX: {
      return cudf::make_max_aggregation();
    }
    case AggregationCode::COUNT: {
      return cudf::make_count_aggregation();
    }
    case AggregationCode::PROD: {
      return cudf::make_product_aggregation();
    }
    case AggregationCode::MEAN: {
      return cudf::make_mean_aggregation();
    }
    case AggregationCode::VAR: {
      return cudf::make_variance_aggregation();
    }
    case AggregationCode::STD: {
      return cudf::make_std_aggregation();
    }
    case AggregationCode::SIZE: {
      return cudf::make_count_aggregation(cudf::null_policy::INCLUDE);
    }
    case AggregationCode::ANY: {
      return cudf::make_any_aggregation();
    }
    case AggregationCode::ALL: {
      return cudf::make_all_aggregation();
    }
    case AggregationCode::SQSUM: {
      return cudf::make_sum_of_squares_aggregation();
    }
    default: {
      assert(false);
      return cudf::make_sum_aggregation();
    }
  }
}

cudf::binary_operator to_cudf_binary_op(AggregationCode code)
{
  switch (code) {
    case AggregationCode::SUM: {
      return cudf::binary_operator::ADD;
    }
    case AggregationCode::MIN: {
      return cudf::binary_operator::NULL_MIN;
    }
    case AggregationCode::MAX: {
      return cudf::binary_operator::NULL_MAX;
    }
    case AggregationCode::PROD: {
      return cudf::binary_operator::MUL;
    }
    default: {
      assert(false);
      return cudf::binary_operator::ADD;
    }
  }
}

cudf::io::compression_type to_cudf_compression(CompressionType compression)
{
  switch (compression) {
    case CompressionType::UNCOMPRESSED: return cudf::io::compression_type::NONE;
    case CompressionType::SNAPPY: return cudf::io::compression_type::SNAPPY;
    case CompressionType::GZIP: return cudf::io::compression_type::GZIP;
    case CompressionType::BROTLI: return cudf::io::compression_type::BROTLI;
    case CompressionType::BZ2: return cudf::io::compression_type::BZIP2;
    case CompressionType::ZIP: return cudf::io::compression_type::ZIP;
    case CompressionType::XZ: return cudf::io::compression_type::XZ;
  }
  return cudf::io::compression_type::AUTO;
}

}  // namespace pandas
}  // namespace legate

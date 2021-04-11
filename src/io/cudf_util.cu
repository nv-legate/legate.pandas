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

#include "io/cudf_util.cuh"

#include <cudf/io/types.hpp>

namespace legate {
namespace pandas {
namespace io {

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

}  // namespace io
}  // namespace pandas
}  // namespace legate

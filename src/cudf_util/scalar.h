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
#include "cudf_util/types.h"
#include "scalar/scalar.h"

#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>

namespace legate {
namespace pandas {

template <TypeCode CODE>
std::unique_ptr<cudf::scalar> to_cudf_scalar(const void *val, cudaStream_t stream)
{
  using T   = pandas_type_of<CODE>;
  auto type = cudf::data_type{to_cudf_type_id(CODE)};
  std::unique_ptr<cudf::scalar> scalar;
  switch (CODE) {
    case TypeCode::TS_NS: {
      scalar = cudf::make_timestamp_scalar(type, stream);
      break;
    }
    default: {
      scalar = cudf::make_numeric_scalar(type, stream);
      break;
    }
  }
  if (nullptr != val)
    static_cast<cudf::scalar_type_t<T> *>(scalar.get())
      ->set_value(*static_cast<const T *>(val), stream);
  return std::move(scalar);
}

template <>
inline std::unique_ptr<cudf::scalar> to_cudf_scalar<TypeCode::STRING>(const void *val,
                                                                      cudaStream_t stream)
{
  if (nullptr != val)
    return cudf::make_string_scalar(*static_cast<const std::string *>(val), stream);
  else
    return std::make_unique<cudf::string_scalar>(std::string{}, false, stream);
}

std::unique_ptr<cudf::scalar> to_cudf_scalar(const void *val, TypeCode code, cudaStream_t stream);

Scalar from_cudf_scalar(TypeCode code, std::unique_ptr<cudf::scalar> &&in, cudaStream_t stream);

}  // namespace pandas
}  // namespace legate

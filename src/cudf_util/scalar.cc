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

#include "cudf_util/scalar.h"
#include "util/type_dispatch.h"

namespace legate {
namespace pandas {

namespace detail {

struct ToCudfScalar {
  template <TypeCode CODE, std::enable_if_t<CODE != TypeCode::CAT32> * = nullptr>
  std::unique_ptr<cudf::scalar> operator()(const void *val, cudaStream_t stream)
  {
    return to_cudf_scalar<CODE>(val, stream);
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::CAT32> * = nullptr>
  std::unique_ptr<cudf::scalar> operator()(const void *val, cudaStream_t stream)
  {
    return nullptr;
  }
};

}  // namespace detail

std::unique_ptr<cudf::scalar> to_cudf_scalar(const void *val, TypeCode code, cudaStream_t stream)
{
  return type_dispatch(code, detail::ToCudfScalar{}, val, stream);
}

struct FromCudfScalar {
  template <TypeCode CODE, std::enable_if_t<is_primitive_type<CODE>::value> * = nullptr>
  Scalar operator()(std::unique_ptr<cudf::scalar> &&in, cudaStream_t stream)
  {
    using VAL = pandas_type_of<CODE>;
    return Scalar(in->is_valid(stream),
                  static_cast<const cudf::scalar_type_t<VAL> *>(in.get())->value(stream));
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::STRING> * = nullptr>
  Scalar operator()(std::unique_ptr<cudf::scalar> &&in, cudaStream_t stream)
  {
    auto valid = in->is_valid(stream);
    if (!valid) return Scalar(TypeCode::STRING);
    auto value = static_cast<const cudf::scalar_type_t<std::string> *>(in.get())->to_string(stream);
    return Scalar(valid, value);
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::CAT32> * = nullptr>
  Scalar operator()(std::unique_ptr<cudf::scalar> &&in, cudaStream_t stream)
  {
    assert(false);
    return Scalar();
  }
};

Scalar from_cudf_scalar(TypeCode code, std::unique_ptr<cudf::scalar> &&in, cudaStream_t stream)
{
  return type_dispatch(code, FromCudfScalar{}, std::move(in), stream);
}

}  // namespace pandas
}  // namespace legate

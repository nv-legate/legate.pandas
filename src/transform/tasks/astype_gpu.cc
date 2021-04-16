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

#include "transform/tasks/astype.h"
#include "category/conversion.h"
#include "column/column.h"
#include "cudf_util/allocators.h"
#include "cudf_util/column.h"
#include "cudf_util/types.h"
#include "util/gpu_task_context.h"
#include "deserializer.h"

#include <cudf/detail/unary.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/detail/converters.hpp>

namespace legate {
namespace pandas {
namespace transform {

using namespace Legion;

namespace detail {

std::unique_ptr<cudf::column> from_string(cudf::data_type result_type,
                                          const cudf::strings_column_view &input,
                                          cudaStream_t stream,
                                          rmm::mr::device_memory_resource *mr)
{
  switch (result_type.id()) {
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::INT64:
    case cudf::type_id::UINT8:
    case cudf::type_id::UINT16:
    case cudf::type_id::UINT32:
    case cudf::type_id::UINT64: {
      return cudf::strings::detail::to_integers(input, result_type, stream, mr);
    }
    case cudf::type_id::FLOAT32:
    case cudf::type_id::FLOAT64: {
      return cudf::strings::detail::to_floats(input, result_type, stream, mr);
    }
    case cudf::type_id::BOOL8: {
      cudf::string_scalar true_string("True", true, stream);
      return cudf::strings::detail::to_booleans(input, true_string, stream, mr);
    }
    case cudf::type_id::DICTIONARY32: {
      // Unreachable
      assert(false);
      break;
    }
    case cudf::type_id::TIMESTAMP_NANOSECONDS: {
      // TODO: Need to add datetime64[ns]-to-string conversion
      assert(false);
      break;
    }
  }
  assert(false);
  return nullptr;
}

std::unique_ptr<cudf::column> to_string(const cudf::column_view &input,
                                        cudaStream_t stream,
                                        rmm::mr::device_memory_resource *mr)
{
  switch (input.type().id()) {
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::INT64:
    case cudf::type_id::UINT8:
    case cudf::type_id::UINT16:
    case cudf::type_id::UINT32:
    case cudf::type_id::UINT64: {
      return cudf::strings::detail::from_integers(input, stream, mr);
    }
    case cudf::type_id::FLOAT32:
    case cudf::type_id::FLOAT64: {
      return cudf::strings::detail::from_floats(input, stream, mr);
    }
    case cudf::type_id::BOOL8: {
      cudf::string_scalar true_string("True", true, stream);
      cudf::string_scalar false_string("False", true, stream);
      return cudf::strings::detail::from_booleans(input, true_string, false_string, stream, mr);
    }
    case cudf::type_id::DICTIONARY32: {
      return category::to_string_column(input, stream, mr);
    }
    case cudf::type_id::TIMESTAMP_NANOSECONDS: {
      return cudf::strings::detail::from_timestamps(input, "%Y-%m-%d %H:%M:%S", stream, mr);
    }
  }
  assert(false);
  return nullptr;
}

}  // namespace detail

/*static*/ void AstypeTask::gpu_variant(const Task *task,
                                        const std::vector<PhysicalRegion> &regions,
                                        Context context,
                                        Runtime *runtime)
{
  Deserializer ctx{task, regions};

  OutputColumn out;
  Column<true> in;

  deserialize(ctx, out);
  deserialize(ctx, in);

  if (in.empty()) {
    out.make_empty(true);
    return;
  }

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  auto input = to_cudf_column(in, stream);

  DeferredBufferAllocator mr;
  std::unique_ptr<cudf::column> result;

  switch (out.code()) {
    case TypeCode::STRING: {
      result = detail::to_string(input, stream, &mr);
      break;
    }
    case TypeCode::CAT32: {
      assert(false);
      break;
    }
    case TypeCode::TS_NS: {
      assert(false);
      break;
    }
    default: {
      auto type_id = cudf::data_type(to_cudf_type_id(out.code()));

      if (in.code() == TypeCode::STRING)
        result = detail::from_string(type_id, input, stream, &mr);
      else
        result = cudf::detail::cast(input, type_id, stream, &mr);
      break;
    }
  }

  from_cudf_column(out, std::move(result), stream, mr);
}

}  // namespace transform
}  // namespace pandas
}  // namespace legate

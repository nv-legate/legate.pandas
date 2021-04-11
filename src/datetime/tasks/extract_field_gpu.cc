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

#include "datetime/tasks/extract_field.h"
#include "column/column.h"
#include "column/device_column.h"
#include "cudf_util/allocators.h"
#include "util/gpu_task_context.h"
#include "deserializer.h"

#include <cudf/datetime.hpp>
#include <cudf/detail/binaryop.hpp>
#include <cudf/detail/datetime.hpp>

namespace legate {
namespace pandas {
namespace datetime {

using namespace Legion;

/*static*/ void ExtractFieldTask::gpu_variant(const Task *task,
                                              const std::vector<PhysicalRegion> &regions,
                                              Context context,
                                              Runtime *runtime)
{
  Deserializer ctx{task, regions};

  DatetimeFieldCode code;
  OutputColumn out;
  Column<true> in;

  deserialize(ctx, code);
  deserialize(ctx, out);
  deserialize(ctx, in);

  if (in.empty()) {
    out.make_empty(true);
    return;
  }

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  auto input = DeviceColumn<true>{in}.to_cudf_column(stream);

  DeferredBufferAllocator mr;

  std::unique_ptr<cudf::column> result;
  switch (code) {
    case DatetimeFieldCode::YEAR: {
      result = cudf::datetime::detail::extract_year(input, stream, &mr);
      break;
    }
    case DatetimeFieldCode::MONTH: {
      result = cudf::datetime::detail::extract_month(input, stream, &mr);
      break;
    }
    case DatetimeFieldCode::DAY: {
      result = cudf::datetime::detail::extract_day(input, stream, &mr);
      break;
    }
    case DatetimeFieldCode::HOUR: {
      result = cudf::datetime::detail::extract_hour(input, stream, &mr);
      break;
    }
    case DatetimeFieldCode::MINUTE: {
      result = cudf::datetime::detail::extract_minute(input, stream, &mr);
      break;
    }
    case DatetimeFieldCode::SECOND: {
      result = cudf::datetime::detail::extract_second(input, stream, &mr);
      break;
    }
    case DatetimeFieldCode::WEEKDAY: {
      auto out     = cudf::datetime::detail::extract_weekday(input, stream);
      auto one     = cudf::numeric_scalar<int16_t>(1);
      auto type_id = cudf::data_type{cudf::type_id::INT16};
      result       = cudf::detail::binary_operation(
        out->view(), one, cudf::binary_operator::SUB, type_id, stream, &mr);
      break;
    }
  }

  DeviceOutputColumn{out}.return_from_cudf_column(mr, result->view(), stream);
}

}  // namespace datetime
}  // namespace pandas
}  // namespace legate

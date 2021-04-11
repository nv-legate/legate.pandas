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

#include "reduction/tasks/scan.h"
#include "column/device_column.h"
#include "cudf_util/detail.h"
#include "cudf_util/types.h"
#include "util/gpu_task_context.h"

#include <cudf/copying.hpp>
#include <cudf/reduction.hpp>
#include <cudf/types.hpp>
#include <cudf/detail/binaryop.hpp>

namespace legate {
namespace pandas {
namespace reduction {

using namespace Legion;

/*static*/ void ScanTask::gpu_variant(const Task *task,
                                      const std::vector<PhysicalRegion> &regions,
                                      Context context,
                                      Runtime *runtime)
{
  Deserializer ctx{task, regions};

  ScanArgs args;
  deserialize(ctx, args);

  if (args.output.empty()) return;

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  auto null_policy = args.skipna ? cudf::null_policy::EXCLUDE : cudf::null_policy::INCLUDE;

  if (args.local) {
    auto input = DeviceColumn<true>{args.input}.to_cudf_column(stream);
    {
      CompositeAllocator mr;
      mr.add<SingletonAllocator>(args.output.raw_column_untyped_write(), args.output.bytes());
      if (args.output.nullable()) {
        if (input.has_nulls())
          mr.add<ProxyAllocator>(args.output.write_bitmask(), stream);
        else
          args.output.write_bitmask().set_all_valid(stream);
      }

      cudf::detail::scan(
        input, to_cudf_agg(args.code), cudf::scan_type::INCLUSIVE, null_policy, stream, &mr);
    }

    if (args.has_buffer) {
      bool copy_data = true;
      bool all_nulls = input.size() == input.null_count();
      bool any_nulls = input.null_count() > 0;

      if (args.output.nullable()) {
        auto buffer_b            = args.write_buffer.raw_bitmask_write();
        Bitmask::AllocType valid = !((args.skipna && all_nulls) || (!args.skipna && any_nulls));
        cudaMemcpyAsync(
          buffer_b, &valid, sizeof(Bitmask::AllocType), cudaMemcpyHostToDevice, stream);
        copy_data = valid;
      }

      if (copy_data) {
        auto output    = static_cast<const int8_t *>(args.output.raw_column_untyped_read());
        auto buffer    = args.write_buffer.raw_column_untyped_write();
        auto elem_size = args.output.elem_size();
        auto offset    = (args.output.num_elements() - 1) * elem_size;

        cudaMemcpyAsync(buffer, output + offset, elem_size, cudaMemcpyDeviceToDevice, stream);
      }
    }
  } else {
    if (task->index_point[0] == 0) return;

    auto default_resource = rmm::mr::get_current_device_resource();

    auto buffer     = DeviceColumn<true>{args.read_buffer}.to_cudf_column(stream);
    auto global_agg = cudf::detail::scan(buffer,
                                         to_cudf_agg(args.code),
                                         cudf::scan_type::INCLUSIVE,
                                         null_policy,
                                         stream,
                                         default_resource);
    {
      auto output    = DeviceColumn<false>{args.output}.to_cudf_column(stream);
      auto has_nulls = output.has_nulls();

      auto rhs = cudf::detail::get_element(
        global_agg->view(), task->index_point[0] - 1, stream, default_resource);
      cudf::column lhs{output, stream, default_resource};

      if (!rhs->is_valid(stream)) {
        if (!args.skipna) args.output.write_bitmask().clear(stream);
      } else {
        auto lhs_view = lhs.view();
        CompositeAllocator mr;
        if (lhs_view.has_nulls() || args.code == AggregationCode::MIN ||
            args.code == AggregationCode::MAX)
          mr.add<DeferredBufferAllocator>();
        mr.add<SingletonAllocator>(args.output.raw_column_untyped_write(), args.output.bytes());

        cudf::detail::binary_operation(
          lhs_view, *rhs, to_cudf_binary_op(args.code), lhs.type(), stream, &mr);
      }
    }
  }
}

}  // namespace reduction
}  // namespace pandas
}  // namespace legate

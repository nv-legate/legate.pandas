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

#include "copy/tasks/fill.h"
#include "column/device_column.h"
#include "cudf_util/scalar.h"
#include "util/gpu_task_context.h"
#include "util/type_dispatch.h"

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/detail/fill.hpp>
#include <cudf/detail/repeat.hpp>

namespace legate {
namespace pandas {
namespace copy {

using namespace Legion;

namespace detail {
namespace gpu {

struct Fill {
  template <TypeCode CODE, std::enable_if_t<is_primitive_type<CODE>::value> * = nullptr>
  void operator()(OutputColumn &out, const Scalar &scalar, int64_t size, cudaStream_t stream)
  {
    using VAL = pandas_type_of<CODE>;

    out.allocate(size);
    if (size == 0) return;

    if (!scalar.valid()) {
      out.bitmask().clear(stream);
      return;
    }

    auto p_scalar = to_cudf_scalar<CODE>(scalar.ptr<VAL>(), stream);
    auto m_out    = DeviceOutputColumn{out}.to_mutable_cudf_column();
    cudf::detail::fill_in_place(m_out, 0, static_cast<cudf::size_type>(size), *p_scalar, stream);
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::STRING> * = nullptr>
  void operator()(OutputColumn &out, const Scalar &scalar, int64_t size, cudaStream_t stream)
  {
    if (size == 0) {
      out.make_empty(true);
      return;
    }

    if (!scalar.valid()) {
      out.allocate(size);
      out.bitmask().clear(stream);
      out.child(0).allocate(size + 1);
      out.child(1).allocate(0);
      cudaMemsetAsync(out.child(0).raw_column<int32_t>(), 0, (size + 1) * sizeof(int32_t), stream);
      return;
    }

    auto value = scalar.value<std::string>();

    cudf::size_type num_offsets = 2;
    cudf::size_type num_chars   = value.size();

    alloc::DeferredBufferAllocator alloc(Memory::Z_COPY_MEM);
    void *buffer = alloc.allocate(sizeof(int32_t) * num_offsets + num_chars);

    int32_t *offsets = static_cast<int32_t *>(buffer);
    int8_t *chars    = static_cast<int8_t *>(buffer) + sizeof(int32_t) * num_offsets;
    offsets[0]       = 0;
    offsets[1]       = num_chars;
    memcpy(chars, value.c_str(), num_chars);

    cudf::column_view offset_view{cudf::data_type{cudf::type_id::INT32}, num_offsets, offsets};
    cudf::column_view char_view{cudf::data_type{cudf::type_id::INT8}, num_chars, chars};
    cudf::column_view to_repeat{
      cudf::data_type{cudf::type_id::STRING}, 1, nullptr, nullptr, 0, 0, {offset_view, char_view}};

    DeferredBufferAllocator mr;
    auto result = cudf::detail::repeat(
      cudf::table_view{{to_repeat}}, static_cast<cudf::size_type>(size), stream, &mr);
    DeviceOutputColumn{out}.return_from_cudf_column(mr, result->view().column(0), stream);
  }

  template <TypeCode CODE, std::enable_if_t<CODE == TypeCode::CAT32> * = nullptr>
  void operator()(OutputColumn &out, const Scalar &scalar, int64_t size, cudaStream_t stream)
  {
    assert(false);
  }
};

}  // namespace gpu
}  // namespace detail

/*static*/ void FillTask::gpu_variant(const Task *task,
                                      const std::vector<PhysicalRegion> &regions,
                                      Context context,
                                      Runtime *runtime)
{
  Deserializer ctx{task, regions};

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  FromFuture<int64_t> volume;
  deserialize(ctx, volume);
  OutputColumn out_column;
  deserialize(ctx, out_column);
  int32_t num_pieces;
  deserialize(ctx, num_pieces);
  Scalar value;
  deserialize(ctx, value);

  int64_t total_size = volume.value();
  int64_t task_id    = task->index_point[0];
  int64_t my_size    = (task_id + 1) * total_size / num_pieces - task_id * total_size / num_pieces;

  type_dispatch(value.code(), detail::gpu::Fill{}, out_column, value, my_size, stream);
}

}  // namespace copy
}  // namespace pandas
}  // namespace legate

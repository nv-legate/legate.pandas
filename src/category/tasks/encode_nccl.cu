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

#include "category/tasks/encode_nccl.h"
#include "category/drop_duplicates.h"
#include "category/encode.h"
#include "cudf_util/allocators.h"
#include "cudf_util/column.h"
#include "nccl/shuffle.h"
#include "nccl/util.h"
#include "util/gpu_task_context.h"

#include <cudf/copying.hpp>
#include <cudf/detail/stream_compaction.hpp>

namespace legate {
namespace pandas {
namespace category {

namespace detail {

struct EncodeNCCLTaskArgs {
  uint32_t num_pieces;
  OutputColumn out;
  OutputColumn dict;
  Column<true> in;
  ncclComm_t *comm;

  friend void deserialize(Deserializer &ctx, EncodeNCCLTaskArgs &args)
  {
    deserialize(ctx, args.num_pieces);
    deserialize(ctx, args.out);
    deserialize(ctx, args.dict);
    deserialize(ctx, args.in);
    deserialize_from_future(ctx, args.comm);
  }
};

}  // namespace detail

using namespace Legion;

/*static*/ void EncodeNCCLTask::gpu_variant(const Legion::Task *task,
                                            const std::vector<Legion::PhysicalRegion> &regions,
                                            Legion::Context context,
                                            Legion::Runtime *runtime)
{
  Deserializer ctx{task, regions};

  detail::EncodeNCCLTaskArgs args;
  deserialize(ctx, args);

  GPUTaskContext gpu_ctx{};
  auto stream = gpu_ctx.stream();

  auto input = to_cudf_column(args.in, stream);

  DeferredBufferAllocator output_mr;
  auto global_dedup = detail::drop_duplicates(cudf::table_view{{input}},
                                              std::vector<cudf::size_type>{0},
                                              cudf::duplicate_keep_option::KEEP_FIRST,
                                              cudf::null_equality::EQUAL,
                                              task->index_point[0],
                                              args.num_pieces,
                                              args.comm,
                                              stream,
                                              &output_mr)
                        ->release();

  std::unique_ptr<cudf::column> dict_column(std::move(global_dedup.front()));
  if (input.has_nulls()) {
    dict_column = std::make_unique<cudf::column>(
      cudf::slice(dict_column->view(), std::vector<cudf::size_type>{1, dict_column->size()})
        .front(),
      stream,
      &output_mr);
  }

  if (!args.in.empty()) {
    DeferredBufferAllocator mr;
    auto result = detail::encode(input, dict_column->view(), stream, &mr);
    from_cudf_column(args.out, std::move(result), stream, mr);
  } else
    args.out.make_empty(true);

  if (task->index_point[0] == 0)
    from_cudf_column(args.dict, std::move(dict_column), stream, output_mr);
  else
    args.dict.make_empty();
}

static void __attribute__((constructor)) register_tasks(void)
{
  EncodeNCCLTask::register_variants();
}

}  // namespace category
}  // namespace pandas
}  // namespace legate

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

#include "udf/eval_udf.h"

namespace legate {
namespace pandas {
namespace udf {

using namespace Legion;

void EvalUDFTask::EvalUDFTaskArgs::sanity_check(void)
{
  for (auto &column : columns) assert(column.shape() == columns[0].shape());
}

/*static*/ void EvalUDFTask::cpu_variant(const Task *task,
                                         const std::vector<PhysicalRegion> &regions,
                                         Context context,
                                         Runtime *runtime)
{
  Deserializer ctx{task, regions};

  EvalUDFTaskArgs args;
  deserialize(ctx, args);

#ifdef DEBUG_PANDAS
  assert(!args.columns.empty());
#endif
  const auto size = args.columns[0].num_elements();

  args.mask.allocate(size);
  if (size == 0) return;

  using UDF = void(void **, size_t);
  auto udf  = reinterpret_cast<UDF *>(args.func_ptr);

  std::vector<void *> udf_args;
  udf_args.push_back(args.mask.raw_column_untyped());
  for (auto &column : args.columns)
    udf_args.push_back(const_cast<void *>(column.raw_column_untyped_read()));
  for (auto &scalar : args.scalars) udf_args.push_back(const_cast<void *>(scalar.rawptr_));

  udf(udf_args.data(), size);

  if (!args.mask.nullable()) return;

  bool initialized = false;
  Bitmask bitmask  = args.mask.bitmask();
  for (auto &column : args.columns) {
    if (!column.nullable()) continue;
    Bitmask to_merge = column.read_bitmask();
    if (initialized)
      intersect_bitmasks(bitmask, bitmask, to_merge);
    else
      to_merge.copy(bitmask);
  }
}

void deserialize(Deserializer &ctx, EvalUDFTask::EvalUDFTaskArgs &args)
{
  deserialize(ctx, args.func_ptr);
  deserialize(ctx, args.mask);
  deserialize(ctx, args.columns);
  deserialize(ctx, args.scalars);

#ifdef DEBUG_PANDAS
  args.sanity_check();
#endif
}

static void __attribute__((constructor)) register_tasks(void) { EvalUDFTask::register_variants(); }

}  // namespace udf
}  // namespace pandas
}  // namespace legate

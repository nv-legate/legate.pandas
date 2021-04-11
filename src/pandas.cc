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

#include "mapper.h"
#include "pandas.h"
#include "proj.h"
#include "reduction_ops.h"
#include "shard.h"
#include "util/type_dispatch.h"

namespace legate {
namespace pandas {

using namespace Legion;

TypeCode to_storage_type_code(TypeCode code)
{
  if (code == TypeCode::TS_NS)
    return TypeCode::INT64;
  else
    return code;
}

bool is_primitive_type_code(TypeCode code)
{
  switch (code) {
    case TypeCode::BOOL:
    case TypeCode::INT8:
    case TypeCode::INT16:
    case TypeCode::INT32:
    case TypeCode::INT64:
    case TypeCode::UINT8:
    case TypeCode::UINT16:
    case TypeCode::UINT32:
    case TypeCode::UINT64:
    case TypeCode::FLOAT:
    case TypeCode::DOUBLE:
    // Timestamps aren't really numeric, but we treat them as numeric values
    case TypeCode::TS_NS: {
      return true;
    }
    case TypeCode::STRING:
    case TypeCode::CAT32: {
      return false;
    }
    default: {
      assert(false);
      return false;
    }
  }
}

namespace detail {

struct ElemSize {
  template <TypeCode CODE, std::enable_if_t<is_primitive_type<CODE>::value> * = nullptr>
  size_t operator()()
  {
    using VAL = pandas_type_of<CODE>;
    return sizeof(VAL);
  }

  template <TypeCode CODE, std::enable_if_t<!is_primitive_type<CODE>::value> * = nullptr>
  size_t operator()()
  {
    return 0;
  }
};

}  // namespace detail

size_t size_of_type(TypeCode code) { return type_dispatch(code, detail::ElemSize{}); }

static const char *const pandas_library_name = "legate.pandas";

/*static*/ void LegatePandas::record_variant(TaskID tid,
                                             const char *task_name,
                                             const CodeDescriptor &descriptor,
                                             ExecutionConstraintSet &execution_constraints,
                                             TaskLayoutConstraintSet &layout_constraints,
                                             VariantID var,
                                             Processor::Kind kind,
                                             bool leaf,
                                             bool inner,
                                             bool idempotent,
                                             bool ret_type)
{
  log_legate.info("Recording %s", task_name);
  assert((kind == Processor::LOC_PROC) || (kind == Processor::TOC_PROC));
  std::deque<PendingTaskVariant> &pending_task_variants = get_pending_task_variants();
  // Buffer these up until we can do our actual registration with the runtime
  pending_task_variants.push_back(PendingTaskVariant(tid,
                                                     false /*global*/,
                                                     (kind == Processor::LOC_PROC) ? "CPU" : "GPU",
                                                     task_name,
                                                     descriptor,
                                                     var,
                                                     ret_type));
  TaskVariantRegistrar &registrar = pending_task_variants.back();
  registrar.execution_constraints.swap(execution_constraints);
  registrar.layout_constraints.swap(layout_constraints);
  registrar.add_constraint(ProcessorConstraint(kind));
  registrar.set_leaf(leaf);
  registrar.set_inner(inner);
  registrar.set_idempotent(idempotent);
  // Everyone is doing registration on their own nodes
  registrar.global_registration = false;
}

/*static*/ std::deque<LegatePandas::PendingTaskVariant> &LegatePandas::get_pending_task_variants(
  void)
{
  static std::deque<PendingTaskVariant> pending_task_variants;
  return pending_task_variants;
}

/*static*/ void LegatePandas::registration_callback(Machine machine,
                                                    Runtime *runtime,
                                                    const std::set<Processor> &local_procs)
{
  log_legate.info("Registration callback invoked");
  // This is the callback that we get from the runtime after it has started
  // but before the actual application starts running so we can now do all
  // our registrations.
  // First let's get our range of task IDs for this library from the runtime
  const size_t max_pandas_tasks = 1 << 20;
  const TaskID first_tid =
    runtime->generate_library_task_ids(pandas_library_name, max_pandas_tasks);
  std::deque<PendingTaskVariant> &pending_task_variants = get_pending_task_variants();
  log_legate.info("# of pending variants: %lu", pending_task_variants.size());
  // Do all our registrations
  for (std::deque<PendingTaskVariant>::iterator it = pending_task_variants.begin();
       it != pending_task_variants.end();
       it++) {
    // Make sure we haven't exceed our maximum range of IDs
    assert(it->task_id < max_pandas_tasks);
    it->task_id += first_tid;  // Add in our library offset
    // Attach the task name too for debugging
    runtime->attach_name(it->task_id, it->task_name, false /*mutable*/, true /*local only*/);
    log_legate.info("Registering %s (tid: %u)", it->task_name, it->task_id);
    runtime->register_task_variant(*it, it->descriptor, NULL, 0, it->ret_type, it->var);
  }
  pending_task_variants.clear();

  const ProjectionID first_pid =
    runtime->generate_library_projection_ids(pandas_library_name, NUM_PROJ);
  projection::PandasProjectionFunctor::register_projection_functors(runtime, first_pid);

  const ReductionOpID first_rid = runtime->generate_library_reduction_ids(pandas_library_name, 1);
  runtime->register_reduction_op<RangeUnion>(first_rid + PANDAS_REDOP_RANGE_UNION);

  const ShardingID first_sid =
    runtime->generate_library_sharding_ids(pandas_library_name, NUM_SHARD);
  sharding::PandasShardingFunctor::register_sharding_functors(runtime, first_sid);

  const MapperID pandas_mapper_id = runtime->generate_library_mapper_ids(pandas_library_name, 1);
  runtime->add_mapper(pandas_mapper_id,
                      new mapper::PandasMapper(runtime->get_mapper_runtime(),
                                               machine,
                                               first_tid,
                                               first_tid + max_pandas_tasks - 1,
                                               first_sid));
  log_legate.info("Registering mapper (id: %u)", pandas_mapper_id);
}

}  // namespace pandas
}  // namespace legate

extern "C" {

void legate_pandas_perform_registration()
{
  // Tell the runtime about our registration callback so we hook it
  // in before the runtime starts
  Legion::Runtime::perform_registration_callback(
    legate::pandas::LegatePandas::registration_callback, true);
}

#ifndef LEGATE_USE_CUDA
unsigned legate_pandas_get_cuda_arch() { return -1U; }
bool legate_pandas_use_nccl() { return false; }
#endif
}

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

#include <algorithm>
#include <unordered_set>

#include "pandas.h"
#include "mapper.h"

namespace legate {
namespace pandas {
namespace mapper {

static const char* mapping_error_message =
  "Failed allocation of size %zd bytes for "
  "region requirement %u of task %s (UID %lld) in memory " IDFMT " for processor " IDFMT
  ". This means the working "
  "set of your application is too big for the allotted "
  "capacity of the given memory under the default "
  "mapper's mapping scheme. You have three choices: "
  "ask Realm to allocate more memory, write a custom "
  "mapper to better manage working sets, or find a bigger "
  "machine.";

static const char* inline_mapping_error_message =
  "Failed allocation of size %zd bytes for "
  "inline mapping (UID %lld) in memory " IDFMT " for processor " IDFMT
  ". This means the working "
  "set of your application is too big for the allotted "
  "capacity of the given memory under the default "
  "mapper's mapping scheme. You have three choices: "
  "ask Realm to allocate more memory, write a custom "
  "mapper to better manage working sets, or find a bigger "
  "machine.";

static const char* deppart_error_message =
  "Failed allocation of size %zd bytes for "
  "dependent partition op (UID %lld) in memory " IDFMT " for processor " IDFMT
  ". This means the working "
  "set of your application is too big for the allotted "
  "capacity of the given memory under the default "
  "mapper's mapping scheme. You have three choices: "
  "ask Realm to allocate more memory, write a custom "
  "mapper to better manage working sets, or find a bigger "
  "machine.";

Logger log_pandas("pandas");

//--------------------------------------------------------------------------
PandasMapper::PandasMapper(
  MapperRuntime* runtime, Machine machine, TaskID first, TaskID last, ShardingID first_shard)
  : NullMapper(runtime, machine),
    first_pandas_task_id(first),
    last_pandas_task_id(last),
    first_pandas_sharding_id(first_shard),
    total_nodes(get_total_nodes(machine))
//--------------------------------------------------------------------------
{
  // Query to find all our local processors
  Machine::ProcessorQuery local_procs(machine);
  local_procs.local_address_space();
  for (Machine::ProcessorQuery::iterator it = local_procs.begin(); it != local_procs.end(); it++) {
    switch (it->kind()) {
      case Processor::LOC_PROC: {
        local_cpus.push_back(*it);
        break;
      }
      case Processor::TOC_PROC: {
        local_gpus.push_back(*it);
        break;
      }
      case Processor::OMP_PROC: {
        local_omps.push_back(*it);
        break;
      }
      case Processor::IO_PROC: {
        local_ios.push_back(*it);
        break;
      }
      case Processor::PY_PROC: {
        local_pys.push_back(*it);
        break;
      }
      default: break;
    }
  }
  // Now do queries to find all our local memories
  Machine::MemoryQuery local_sysmem(machine);
  local_sysmem.local_address_space();
  local_sysmem.only_kind(Memory::SYSTEM_MEM);
  assert(local_sysmem.count() > 0);
  local_system_memory = local_sysmem.first();
  if (!local_gpus.empty()) {
    Machine::MemoryQuery local_zcmem(machine);
    local_zcmem.local_address_space();
    local_zcmem.only_kind(Memory::Z_COPY_MEM);
    assert(local_zcmem.count() > 0);
    local_zerocopy_memory = local_zcmem.first();
  }
  for (std::vector<Processor>::const_iterator it = local_gpus.begin(); it != local_gpus.end();
       it++) {
    Machine::MemoryQuery local_framebuffer(machine);
    local_framebuffer.local_address_space();
    local_framebuffer.only_kind(Memory::GPU_FB_MEM);
    local_framebuffer.best_affinity_to(*it);
    assert(local_framebuffer.count() > 0);
    local_frame_buffers[*it] = local_framebuffer.first();
  }
  for (std::vector<Processor>::const_iterator it = local_omps.begin(); it != local_omps.end();
       it++) {
    Machine::MemoryQuery local_numa(machine);
    local_numa.local_address_space();
    local_numa.only_kind(Memory::SOCKET_MEM);
    local_numa.best_affinity_to(*it);
    if (local_numa.count() > 0)  // if we have NUMA memories then use them
      local_numa_domains[*it] = local_numa.first();
    else  // Otherwise we just use the local system memory
      local_numa_domains[*it] = local_system_memory;
  }
}

//--------------------------------------------------------------------------
PandasMapper::~PandasMapper(void)
//--------------------------------------------------------------------------
{
}

//--------------------------------------------------------------------------
const char* PandasMapper::get_mapper_name(void) const
//--------------------------------------------------------------------------
{
  return "pandas_mapper";
}

//--------------------------------------------------------------------------
Mapper::MapperSyncModel PandasMapper::get_mapper_sync_model(void) const
//--------------------------------------------------------------------------
{
  return CONCURRENT_MAPPER_MODEL;
}

//--------------------------------------------------------------------------
void PandasMapper::configure_context(const MapperContext ctx,
                                     const Task& task,
                                     ContextConfigOutput& output)
//--------------------------------------------------------------------------
{
}

//--------------------------------------------------------------------------
void PandasMapper::select_tasks_to_map(const MapperContext ctx,
                                       const SelectMappingInput& input,
                                       SelectMappingOutput& output)
//--------------------------------------------------------------------------
{
  for (auto const& task : input.ready_tasks) output.map_tasks.insert(task);
}

//--------------------------------------------------------------------------
void PandasMapper::select_steal_targets(const MapperContext ctx,
                                        const SelectStealingInput& input,
                                        SelectStealingOutput& output)
//--------------------------------------------------------------------------
{
}

//--------------------------------------------------------------------------
void PandasMapper::permit_steal_request(const MapperContext ctx,
                                        const StealRequestInput& input,
                                        StealRequestOutput& output)
//--------------------------------------------------------------------------
{
}

//--------------------------------------------------------------------------
void PandasMapper::select_sharding_functor(const Mapping::MapperContext ctx,
                                           const Task& task,
                                           const SelectShardingFunctorInput& input,
                                           SelectShardingFunctorOutput& output)
//--------------------------------------------------------------------------
{
  output.chosen_functor = first_pandas_sharding_id + static_cast<ShardID>(ShardingCode::SHARD_TILE);
}

//--------------------------------------------------------------------------
void PandasMapper::map_future_map_reduction(const MapperContext ctx,
                                            const FutureMapReductionInput& input,
                                            FutureMapReductionOutput& output)
//--------------------------------------------------------------------------
{
  // Don't need to do anything for now
}

//--------------------------------------------------------------------------
void PandasMapper::select_task_options(const MapperContext ctx,
                                       const Task& task,
                                       TaskOptions& output)
//--------------------------------------------------------------------------
{
  Processor::Kind preferred_kind =
    local_gpus.size() > 0 ? Processor::TOC_PROC : Processor::LOC_PROC;

  std::vector<VariantID> variants;
  runtime->find_valid_variants(ctx, task.task_id, variants, preferred_kind);

  if (variants.size() == 0) preferred_kind = Processor::LOC_PROC;

  switch (preferred_kind) {
    case Processor::LOC_PROC: {
      output.initial_proc = *local_cpus.begin();
      break;
    }
    case Processor::TOC_PROC: {
      output.initial_proc = *local_gpus.begin();
      break;
    }
    default: {
      assert(false);
      break;
    }
  }

  output.inline_task = false;
  output.stealable   = false;
  output.map_locally = false;

#ifdef DEBUG_PANDAS
  // We shouldn't be mapping a top-level task in this mapper
  assert(task.get_depth() > 0);
#endif
  output.replicate = false;
}

//--------------------------------------------------------------------------
void PandasMapper::slice_task(const MapperContext ctx,
                              const Task& task,
                              const SliceTaskInput& input,
                              SliceTaskOutput& output)
//--------------------------------------------------------------------------
{
  std::vector<Processor>* pprocs = NULL;
  switch (task.target_proc.kind()) {
    case Processor::LOC_PROC: {
      pprocs = &local_cpus;
      break;
    }
    case Processor::TOC_PROC: {
      pprocs = &local_gpus;
      break;
    }
    default: {
      assert(false);
      break;
    }
  }
#ifdef DEBUG_PANDAS
  assert(pprocs != NULL);
#endif

  size_t idx = 0;
  const Rect<1> rect(input.domain);
  auto const& procs = *pprocs;
  for (PointInRectIterator<1> pir(rect); pir(); ++pir, ++idx) {
    Rect<1> slice(*pir, *pir);
    output.slices.emplace_back(slice, procs[idx % procs.size()], false, false);
  }
}

//--------------------------------------------------------------------------
void PandasMapper::map_task(const MapperContext ctx,
                            const Task& task,
                            const MapTaskInput& input,
                            MapTaskOutput& output)
//--------------------------------------------------------------------------
{
  output.task_priority = 0;
  output.postmap_task  = false;
  output.target_procs.push_back(task.target_proc);

  Memory target_memory;
  switch (task.target_proc.kind()) {
    case LOC_PROC: {
      output.chosen_variant = LEGATE_CPU_VARIANT;
      target_memory         = local_system_memory;
      break;
    }
    case TOC_PROC: {
      output.chosen_variant = LEGATE_GPU_VARIANT;
      target_memory         = local_frame_buffers[task.target_proc];
      break;
    }
    default: {
      assert(false);
      break;
    }
  }
#ifdef DEBUG_PANDAS
  assert(target_memory.exists());
#endif

  // Build constraints for validity checking
  LayoutConstraintSet constraints;
  constraints.add_constraint(MemoryConstraint{target_memory.kind()})
    .add_constraint(OrderingConstraint{{DIM_X, DIM_F}, false})
    .add_constraint(FieldConstraint{false, false});

  for (unsigned idx = 0; idx < task.regions.size(); idx++) {
    auto const& req = task.regions[idx];
    auto& in        = input.valid_instances[idx];
    auto& out       = output.chosen_instances[idx];

    if ((req.privilege == NO_ACCESS) || req.privilege_fields.empty()) continue;

#ifdef DEBUG_PANDAS
    if (req.privilege == REDUCE) assert(false);
#endif

    // Handle histograms specially
    if (req.tag == HISTOGRAM) {
#ifdef DEBUG_PANDAS
      assert(req.is_no_access());
#endif
      LayoutConstraintSet inst_constraints;
      inst_constraints.add_constraint(SpecializedConstraint{NORMAL_SPECIALIZE, 0, false, true})
        .add_constraint(MemoryConstraint{local_system_memory.kind()})
        .add_constraint(OrderingConstraint{{DIM_X, DIM_Y, DIM_F}, false})
        .add_constraint(FieldConstraint{req.privilege_fields, false, false});

      PhysicalInstance instance;
      size_t footprint = 0;
      bool created     = false;
      if (!runtime->find_or_create_physical_instance(ctx,
                                                     local_system_memory,
                                                     inst_constraints,
                                                     {req.region},
                                                     instance,
                                                     created,
                                                     true,
                                                     0,
                                                     true,
                                                     &footprint))
        log_pandas.error(mapping_error_message,
                         footprint,
                         idx,
                         task.get_task_name(),
                         task.get_unique_id(),
                         target_memory.id,
                         task.target_proc.id);

#ifdef DEBUG_PANDAS
      const LayoutConstraint* failed_constraint = NULL;
      bool entails = instance.entails(inst_constraints, &failed_constraint);
      assert(entails);
#endif

      if (created) runtime->set_garbage_collection_priority(ctx, instance, GC_FIRST_PRIORITY);
      out.push_back(instance);
      continue;
    }

    Domain domain{runtime->get_index_space_domain(ctx, req.region.get_index_space())};
    std::set<FieldID> fields_to_map(req.privilege_fields.begin(), req.privilege_fields.end());

    for (auto& inst : in) {
      assert(inst.is_normal_instance());
      std::set<FieldID> inst_fields;
      inst.get_fields(inst_fields);

      std::vector<FieldID> fields_to_check;
      std::set_intersection(fields_to_map.begin(),
                            fields_to_map.end(),
                            inst_fields.begin(),
                            inst_fields.end(),
                            std::back_inserter(fields_to_check));

      if (domain != inst.get_instance_domain()) continue;

      // For now we ignore partially satisfied cases
      if (inst.entails(constraints) && runtime->acquire_instance(ctx, inst)) {
        out.push_back(inst);
        for (auto const& fid : fields_to_check) fields_to_map.erase(fid);
      }

      if (fields_to_map.empty()) break;
    }

    if (!fields_to_map.empty()) {
      LayoutConstraintSet inst_constraints = constraints;
      if (domain.dense())
        inst_constraints.add_constraint(
          SpecializedConstraint{LEGION_AFFINE_SPECIALIZE, 0, false, true});
      else
        inst_constraints.add_constraint(
          SpecializedConstraint{LEGION_COMPACT_SPECIALIZE, 0, false, true});
      inst_constraints.add_constraint(FieldConstraint{fields_to_map, false, false});

      PhysicalInstance instance;
      size_t footprint = 0;
      bool created     = false;
      if (!runtime->find_or_create_physical_instance(ctx,
                                                     target_memory,
                                                     inst_constraints,
                                                     {req.region},
                                                     instance,
                                                     created,
                                                     true,
                                                     GC_FIRST_PRIORITY,
                                                     true,
                                                     &footprint))
        log_pandas.error(mapping_error_message,
                         footprint,
                         idx,
                         task.get_task_name(),
                         task.get_unique_id(),
                         target_memory.id,
                         task.target_proc.id);

#ifdef DEBUG_PANDAS
      const LayoutConstraint* failed_constraint = NULL;
      bool entails = instance.entails(inst_constraints, &failed_constraint);
      assert(entails);
#endif

      out.push_back(instance);
    }
  }
  for (auto& output_target : output.output_targets) output_target = target_memory;
}

//--------------------------------------------------------------------------
void PandasMapper::select_task_sources(const MapperContext ctx,
                                       const Task& task,
                                       const SelectTaskSrcInput& input,
                                       SelectTaskSrcOutput& output)
//--------------------------------------------------------------------------
{
  using Rank = std::pair<PhysicalInstance, unsigned>;

  auto& sources = input.source_instances;
  auto& ranking = output.chosen_ranking;

  std::map<Memory, unsigned> source_memories;
  Memory destination_memory = input.target.get_location();
  std::vector<MemoryMemoryAffinity> affinity(1);
  std::vector<Rank> band_ranking(sources.size());
  for (unsigned idx = 0; idx < sources.size(); idx++) {
    const PhysicalInstance& instance = sources[idx];
    Memory location                  = instance.get_location();
    auto finder                      = source_memories.find(location);
    if (finder == source_memories.end()) {
      affinity.clear();
      machine.get_mem_mem_affinity(affinity, location, destination_memory, false);
      unsigned memory_bandwidth = 0;
      if (!affinity.empty()) {
        assert(affinity.size() == 1);
        memory_bandwidth = affinity[0].bandwidth;
      }
      source_memories[location] = memory_bandwidth;
      band_ranking[idx]         = Rank{instance, memory_bandwidth};
    } else
      band_ranking[idx] = Rank{instance, finder->second};
  }

  auto physical_sort_func = [](const Rank& left, const Rank& right) {
    return left.second < right.second;
  };
  std::sort(band_ranking.begin(), band_ranking.end(), physical_sort_func);

  for (auto rank : band_ranking) ranking.push_back(rank.first);
}

//--------------------------------------------------------------------------
void PandasMapper::map_inline(const MapperContext ctx,
                              const InlineMapping& inline_op,
                              const MapInlineInput& input,
                              MapInlineOutput& output)
//--------------------------------------------------------------------------
{
  auto proc = inline_op.parent_task->current_proc;
  Memory target_memory;
  switch (proc.kind()) {
    case LOC_PROC: {
      target_memory = local_system_memory;
      break;
    }
    case TOC_PROC: {
      target_memory = local_frame_buffers[proc];
      break;
    }
    default: {
      assert(false);
      break;
    }
  }

  const RegionRequirement& req = inline_op.requirement;
  LayoutConstraintSet constraints;
  constraints.add_constraint(SpecializedConstraint{NORMAL_SPECIALIZE, 0, false, true})
    .add_constraint(MemoryConstraint{target_memory.kind()})
    .add_constraint(OrderingConstraint{{DIM_X, DIM_F}, false})
    .add_constraint(FieldConstraint{req.privilege_fields, false, false});

  PhysicalInstance instance;
  size_t footprint;
  if (!runtime->create_physical_instance(ctx,
                                         target_memory,
                                         constraints,
                                         {req.region},
                                         instance,
                                         true,
                                         GC_FIRST_PRIORITY,
                                         true,
                                         &footprint))
    log_pandas.error(inline_mapping_error_message,
                     footprint,
                     inline_op.get_unique_id(),
                     target_memory.id,
                     inline_op.parent_task->current_proc.id);
  output.chosen_instances.push_back(instance);
}

//--------------------------------------------------------------------------
void PandasMapper::select_partition_projection(const MapperContext ctx,
                                               const Partition& partition,
                                               const SelectPartitionProjectionInput& input,
                                               SelectPartitionProjectionOutput& output)
//--------------------------------------------------------------------------
{
  if (!input.open_complete_partitions.empty())
    output.chosen_partition = input.open_complete_partitions[0];
  else
    output.chosen_partition = LogicalPartition::NO_PART;
}

//--------------------------------------------------------------------------
void PandasMapper::map_partition(const MapperContext ctx,
                                 const Partition& partition,
                                 const MapPartitionInput& input,
                                 MapPartitionOutput& output)
//--------------------------------------------------------------------------
{
  output.chosen_instances = input.valid_instances;
  if (!output.chosen_instances.empty())
    runtime->acquire_and_filter_instances(ctx, output.chosen_instances);

  std::vector<unsigned> to_erase;
  std::set<FieldID> missing_fields = partition.requirement.privilege_fields;
  for (std::vector<PhysicalInstance>::const_iterator it = output.chosen_instances.begin();
       it != output.chosen_instances.end();
       it++) {
    if (it->get_location().kind() == Memory::GPU_FB_MEM)
      to_erase.push_back(it - output.chosen_instances.begin());
    else {
      it->remove_space_fields(missing_fields);
      if (missing_fields.empty()) break;
    }
  }

  for (std::vector<unsigned>::const_reverse_iterator it = to_erase.rbegin(); it != to_erase.rend();
       it++)
    output.chosen_instances.erase((*it) + output.chosen_instances.begin());

  if (missing_fields.empty()) return;

  LayoutConstraintSet inst_constraints;
  inst_constraints.add_constraint(SpecializedConstraint{NORMAL_SPECIALIZE, 0, false, true})
    .add_constraint(MemoryConstraint{local_system_memory.kind()})
    .add_constraint(OrderingConstraint{{DIM_X, DIM_F}, false})
    .add_constraint(FieldConstraint{missing_fields, false, false});

  PhysicalInstance instance;
  size_t footprint = 0;
  if (!runtime->create_physical_instance(ctx,
                                         local_system_memory,
                                         inst_constraints,
                                         {partition.requirement.region},
                                         instance,
                                         true,
                                         GC_FIRST_PRIORITY,
                                         true,
                                         &footprint))
    log_pandas.error(deppart_error_message,
                     footprint,
                     partition.get_unique_id(),
                     local_system_memory.id,
                     partition.parent_task->current_proc.id);
  output.chosen_instances.push_back(instance);
}

//--------------------------------------------------------------------------
void PandasMapper::select_sharding_functor(const MapperContext ctx,
                                           const Partition& partition,
                                           const SelectShardingFunctorInput& input,
                                           SelectShardingFunctorOutput& output)
//--------------------------------------------------------------------------
{
  output.chosen_functor = first_pandas_sharding_id + static_cast<ShardID>(ShardingCode::SHARD_TILE);
}

//--------------------------------------------------------------------------
void PandasMapper::select_tunable_value(const MapperContext ctx,
                                        const Task& task,
                                        const SelectTunableInput& input,
                                        SelectTunableOutput& output)
//--------------------------------------------------------------------------
{
  switch (input.tunable_id) {
    case PANDAS_TUNABLE_NUM_PIECES: {
      if (!local_gpus.empty()) {  // If we have GPUs, use those
        pack_tunable(local_gpus.size() * total_nodes, output);
      } else if (!local_omps.empty()) {  // Otherwise use OpenMP procs
        pack_tunable(local_omps.size() * total_nodes, output);
      } else {  // Otherwise use the CPUs
        pack_tunable(local_cpus.size() * total_nodes, output);
      }
      break;
    }
    case PANDAS_TUNABLE_HAS_GPUS: {
#ifdef PANDAS_NO_CUDA
      pack_tunable(false, output);
#else
      pack_tunable(local_gpus.size() > 0, output);
#endif
      break;
    }
    default: {
      fprintf(stderr, "Unsupported tunable id: %d\n", input.tunable_id);
      exit(-1);
    }
  }
}

//--------------------------------------------------------------------------
/*static*/ void PandasMapper::pack_tunable(const int value, Mapper::SelectTunableOutput& output)
//--------------------------------------------------------------------------
{
  int* result  = (int*)malloc(sizeof(value));
  *result      = value;
  output.value = result;
  output.size  = sizeof(value);
}

//--------------------------------------------------------------------------
/*static*/ size_t PandasMapper::get_total_nodes(Machine m)
//--------------------------------------------------------------------------
{
  Machine::ProcessorQuery query(m);
  query.only_kind(Processor::LOC_PROC);
  std::unordered_set<AddressSpace> spaces;
  for (Machine::ProcessorQuery::iterator it = query.begin(); it != query.end(); it++)
    spaces.insert(it->address_space());
  return spaces.size();
}

}  // namespace mapper
}  // namespace pandas
}  // namespace legate

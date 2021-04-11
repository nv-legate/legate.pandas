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

#include "legion.h"
#include "pandas.h"
#include "mappers/null_mapper.h"

namespace legate {
namespace pandas {
namespace mapper {

using namespace Legion;
using namespace Legion::Mapping;

class PandasMapper : public NullMapper {
 public:
  enum TunableCode {
    PANDAS_TUNABLE_NUM_PIECES = 1,
    PANDAS_TUNABLE_HAS_GPUS   = 2,
  };
  enum MappingTag {
    BITMASK   = 100,
    HISTOGRAM = 101,
  };

 public:
  PandasMapper(
    MapperRuntime* runtime, Machine machine, TaskID first, TaskID last, ShardingID first_shard);
  virtual ~PandasMapper(void);

 public:
  virtual const char* get_mapper_name(void) const;
  virtual MapperSyncModel get_mapper_sync_model(void) const;

 public:
  virtual void configure_context(const MapperContext ctx,
                                 const Task& task,
                                 ContextConfigOutput& output);

 public:
  virtual void select_tasks_to_map(const MapperContext ctx,
                                   const SelectMappingInput& input,
                                   SelectMappingOutput& output);
  virtual void select_steal_targets(const MapperContext ctx,
                                    const SelectStealingInput& input,
                                    SelectStealingOutput& output);
  virtual void permit_steal_request(const MapperContext ctx,
                                    const StealRequestInput& intput,
                                    StealRequestOutput& output);

 public:
  virtual void select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                       const Legion::Task& task,
                                       const SelectShardingFunctorInput& input,
                                       SelectShardingFunctorOutput& output);

 public:
  virtual void map_future_map_reduction(const MapperContext ctx,
                                        const FutureMapReductionInput& input,
                                        FutureMapReductionOutput& output);

 public:
  virtual void select_task_options(const MapperContext ctx, const Task& task, TaskOptions& output);
  virtual void slice_task(const MapperContext ctx,
                          const Task& task,
                          const SliceTaskInput& input,
                          SliceTaskOutput& output);
  virtual void map_task(const MapperContext ctx,
                        const Task& task,
                        const MapTaskInput& input,
                        MapTaskOutput& output);
  virtual void select_task_sources(const MapperContext ctx,
                                   const Task& task,
                                   const SelectTaskSrcInput& input,
                                   SelectTaskSrcOutput& output);

 public:
  virtual void map_inline(const MapperContext ctx,
                          const InlineMapping& inline_op,
                          const MapInlineInput& input,
                          MapInlineOutput& output);

 public:
  virtual void select_partition_projection(const MapperContext ctx,
                                           const Partition& partition,
                                           const SelectPartitionProjectionInput& input,
                                           SelectPartitionProjectionOutput& output);
  virtual void map_partition(const MapperContext ctx,
                             const Partition& partition,
                             const MapPartitionInput& input,
                             MapPartitionOutput& output);
  virtual void select_sharding_functor(const MapperContext ctx,
                                       const Partition& partition,
                                       const SelectShardingFunctorInput& input,
                                       SelectShardingFunctorOutput& output);

 public:
  virtual void select_tunable_value(const MapperContext ctx,
                                    const Task& task,
                                    const SelectTunableInput& input,
                                    SelectTunableOutput& output);

 protected:
  static void pack_tunable(const int value, Mapper::SelectTunableOutput& output);
  static size_t get_total_nodes(Machine m);

 private:
  TaskID first_pandas_task_id;
  TaskID last_pandas_task_id;
  ShardID first_pandas_sharding_id;
  size_t total_nodes;

 protected:
  std::vector<Processor> local_cpus;
  std::vector<Processor> local_gpus;
  std::vector<Processor> local_omps;  // OpenMP processors
  std::vector<Processor> local_ios;   // I/O processors
  std::vector<Processor> local_pys;   // Python processors
 protected:
  Memory local_system_memory, local_zerocopy_memory;
  std::map<Processor, Memory> local_frame_buffers;
  std::map<Processor, Memory> local_numa_domains;
};

}  // namespace mapper
}  // namespace pandas
}  // namespace legate

# Copyright 2021 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from legate.core import Rect, legion

from legate.pandas.common import types as ty
from legate.pandas.config import OpCode, PandasMappingTag

from .pattern import Map, Projection


class HashPartitioner(object):
    def __init__(self, runtime):
        self._runtime = runtime

    def _hash_partition_cpu(self, columns, key_indices, needs_conversion):
        storage = self._runtime.create_storage(columns[0].ispace)
        out_columns = storage.create_isomorphic_columns(columns)

        _key_indices = list(key_indices)
        for idx in needs_conversion:
            _key_indices[key_indices.index(idx)] = len(columns)
            columns.append(columns[idx].astype(ty.string))
        key_indices = _key_indices

        num_pieces = columns[0].num_pieces
        launch_domain = columns[0].launch_domain
        cspace = columns[0].cspace

        hist_ispace = self._runtime.find_or_create_index_space(
            Rect([num_pieces, num_pieces])
        )
        hist_storage = self._runtime.create_storage(hist_ispace)
        hist = hist_storage.create_new_field(ty.range64)
        hist_ipart = self._runtime.create_row_partition(
            hist_ispace, cspace, num_pieces
        )

        plan = Map(self._runtime, OpCode.LOCAL_PARTITION)

        plan.add_output(
            hist,
            Projection(hist_ipart),
            tag=PandasMappingTag.HISTOGRAM,
            flags=2,  # LEGION_NO_ACCESS_FLAG
        )

        plan.add_scalar_arg(num_pieces, ty.uint32)

        plan.add_scalar_arg(len(key_indices), ty.uint32)
        for idx in key_indices:
            plan.add_scalar_arg(idx, ty.int32)

        plan.add_scalar_arg(len(columns), ty.uint32)
        for key in columns:
            key.add_to_plan(plan, True)
        plan.add_scalar_arg(len(out_columns), ty.uint32)
        for key in out_columns:
            key.add_to_plan_output_only(plan)

        plan.execute(launch_domain)
        del plan

        hist_ipart = self._runtime.create_column_partition(
            hist_ispace, cspace, num_pieces
        )
        radix_ipart = self._runtime.create_partition_by_image(
            columns[0].ispace,
            cspace,
            hist,
            hist_ipart,
            kind=legion.DISJOINT_COMPLETE_KIND,
            range=True,
        )

        out_columns = [
            out_column.all_to_ranges().clone() for out_column in out_columns
        ]
        for out_column in out_columns:
            out_column.set_primary_ipart(radix_ipart)
        out_columns = [
            out_column.all_to_offsets() for out_column in out_columns
        ]

        return out_columns

    def _hash_partition_nccl(self, columns, key_indices, needs_conversion):
        result_storage = self._runtime.create_output_storage()
        out_columns = result_storage.create_similar_columns(columns)

        _key_indices = list(key_indices)
        for idx in needs_conversion:
            _key_indices[key_indices.index(idx)] = len(columns)
            columns.append(columns[idx].astype(ty.string))
        key_indices = _key_indices

        num_pieces = columns[0].num_pieces
        launch_domain = columns[0].launch_domain

        plan = Map(self._runtime, OpCode.GLOBAL_PARTITION)

        plan.add_scalar_arg(num_pieces, ty.uint32)

        plan.add_scalar_arg(len(key_indices), ty.uint32)
        for idx in key_indices:
            plan.add_scalar_arg(idx, ty.int32)

        plan.add_scalar_arg(len(columns), ty.uint32)
        for key in columns:
            key.add_to_plan(plan, True)
        plan.add_scalar_arg(len(out_columns), ty.uint32)
        for key in out_columns:
            key.add_to_plan_output_only(plan)

        plan.add_future_map(self._runtime._nccl_comm)

        self._runtime.issue_fence()
        counts = plan.execute(launch_domain)
        self._runtime.issue_fence()

        result_storage = plan.promote_output_storage(result_storage)
        self._runtime.register_external_weighted_partition(
            result_storage.default_ipart, counts
        )
        del plan

        return out_columns

    def _hash_partition(self, columns, key_indices, needs_conversion=[]):
        if self._runtime.use_nccl:
            return self._hash_partition_nccl(
                columns, key_indices, needs_conversion
            )
        else:
            if not self._runtime.has_gpus:
                needs_conversion = []
            return self._hash_partition_cpu(
                columns, key_indices, needs_conversion
            )

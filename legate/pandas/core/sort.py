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

from legate.pandas.common import types as ty, util as util
from legate.pandas.config import OpCode, PandasMappingTag

from .pattern import Map, Projection


class Sorter(object):
    def __init__(
        self, runtime, frame, by, levels, ascending, na_position, ignore_index
    ):
        self._runtime = runtime
        self._frame = frame
        self._ascending = ascending
        self._put_null_first = na_position == "first"
        self._ignore_index = ignore_index
        self._sort_index = by is None
        self._levels = levels

        if self._sort_index:
            key_start = self._frame.num_columns()
            self._key_indices = [key_start + lvl for lvl in levels]
        else:
            self._key_indices = by

        self._launch_domain = self._frame._columns[0].launch_domain
        self._num_pieces = self._frame._columns[0].num_pieces

    def _prepare_columns(self):
        input_columns = self._frame._columns.copy()

        if not self._ignore_index or self._sort_index:
            input_columns += util.to_list_if_scalar(self._frame._index.column)

        return input_columns

    def _sort(self, renumber_output=False):
        rt = self._runtime

        if renumber_output:
            result_storage = rt.create_output_storage()
            result_columns = result_storage.create_similar_columns(
                self._input_columns
            )
        else:
            result_storage = self._input_columns[0].storage
            result_columns = result_storage.create_isomorphic_columns(
                self._input_columns
            )

        plan = Map(rt, OpCode.SORT_VALUES)

        plan.add_scalar_arg(renumber_output, ty.bool)
        plan.add_scalar_arg(self._put_null_first, ty.bool)
        plan.add_scalar_arg(len(self._key_indices), ty.uint32)
        for asc in self._ascending:
            plan.add_scalar_arg(asc, ty.bool)
        for idx in self._key_indices:
            plan.add_scalar_arg(idx, ty.uint32)
        plan.add_scalar_arg(len(self._input_columns), ty.uint32)
        for column in self._input_columns:
            column.add_to_plan(plan, True)
        for column in result_columns:
            column.add_to_plan_output_only(plan)

        counts = plan.execute(self._launch_domain)

        if renumber_output:
            result_storage = plan.promote_output_storage(result_storage)
            rt.register_external_weighted_partition(
                result_storage.default_ipart, counts.cast(ty.int32)
            )
        del plan

        return result_columns

    def _sample_keys(self):
        rt = self._runtime

        key_columns = [self._input_columns[idx] for idx in self._key_indices]

        sample_storage = rt.create_output_storage()
        sample_columns = sample_storage.create_similar_columns(key_columns)

        plan = Map(rt, OpCode.SAMPLE_KEYS)
        plan.add_scalar_arg(len(key_columns), ty.uint32)
        for column in key_columns:
            column.add_to_plan(plan, True)
        for column in sample_columns:
            column.add_to_plan_output_only(plan)

        plan.execute(self._launch_domain)
        del plan

        return (key_columns, sample_columns)

    def _shuffle_columns(self):
        (self._key_columns, self._sample_columns) = self._sample_keys()

        rt = self._runtime

        cspace = self._input_columns[0].cspace

        hist_ispace = rt.find_or_create_index_space(
            Rect([self._num_pieces, self._num_pieces])
        )
        hist_storage = rt.create_storage(hist_ispace)
        hist = hist_storage.create_new_field(ty.range64)
        hist_ipart = rt.create_row_partition(
            hist_ispace, cspace, self._num_pieces
        )

        # Build histogram using samples. Each point task
        # gets the whole set of samples and sorts them independently.
        plan = Map(rt, OpCode.BUILD_HISTOGRAM)

        plan.add_scalar_arg(self._num_pieces, ty.uint32)
        plan.add_scalar_arg(self._put_null_first, ty.bool)
        plan.add_scalar_arg(len(self._key_columns), ty.uint32)
        for asc in self._ascending:
            plan.add_scalar_arg(asc, ty.bool)
        # Need to broadcast the whole sample region
        samples = [sample.repartition(1) for sample in self._sample_columns]
        for column in samples:
            column.add_to_plan(plan, True, proj=None)
        for column in self._key_columns:
            column.add_to_plan(plan, True)

        plan.add_output(
            hist,
            Projection(hist_ipart),
            tag=PandasMappingTag.HISTOGRAM,
            flags=2,  # LEGION_NO_ACCESS_FLAG
        )

        plan.execute(self._launch_domain)
        del plan

        hist_ipart = rt.create_column_partition(
            hist_ispace, cspace, self._num_pieces
        )
        radix_ipart = rt.create_partition_by_image(
            self._input_columns[0].ispace,
            cspace,
            hist,
            hist_ipart,
            kind=legion.DISJOINT_COMPLETE_KIND,
            range=True,
        )

        # Change the primary partitions to shuffle the data
        input_columns = [
            column.all_to_ranges().clone() for column in self._input_columns
        ]
        for column in input_columns:
            column.set_primary_ipart(radix_ipart)
        input_columns = [column.all_to_offsets() for column in input_columns]
        return input_columns

    def _sort_nccl(self):
        rt = self._runtime
        result_storage = rt.create_output_storage()
        result_columns = result_storage.create_similar_columns(
            self._input_columns
        )

        plan = Map(rt, OpCode.SORT_VALUES_NCCL)

        plan.add_future(self._frame._index.volume)
        plan.add_scalar_arg(self._num_pieces, ty.int32)
        plan.add_scalar_arg(self._put_null_first, ty.bool)
        plan.add_scalar_arg(len(self._key_indices), ty.uint32)
        for asc in self._ascending:
            plan.add_scalar_arg(asc, ty.bool)
        for idx in self._key_indices:
            plan.add_scalar_arg(idx, ty.uint32)
        plan.add_scalar_arg(len(self._input_columns), ty.uint32)
        for column in self._input_columns:
            column.add_to_plan(plan, True)
        for column in result_columns:
            column.add_to_plan_output_only(plan)

        plan.add_future_map(self._runtime._nccl_comm)

        counts = plan.execute(self._launch_domain)

        result_storage = plan.promote_output_storage(result_storage)
        rt.register_external_weighted_partition(
            result_storage.default_ipart, counts.cast(ty.int64)
        )
        del plan

        return result_columns

    def _construct_sort_output(self):
        if self._ignore_index:
            index_columns = []
            value_columns = self._result_columns[: self._frame.num_columns()]
        else:
            nlevels = self._frame._index.nlevels
            index_columns = self._result_columns[-nlevels:]
            value_columns = self._result_columns[:-nlevels]

        return (index_columns, value_columns)

    def sort_values(self):
        self._input_columns = self._prepare_columns()

        if self._num_pieces == 1:
            self._result_columns = self._sort(renumber_output=False)
        elif self._runtime.use_nccl:
            self._runtime.issue_fence()
            self._result_columns = self._sort_nccl()
            self._runtime.issue_fence()
        else:
            self._input_columns = self._sort(renumber_output=False)
            self._input_columns = self._shuffle_columns()
            self._result_columns = self._sort(renumber_output=True)

        return self._construct_sort_output()

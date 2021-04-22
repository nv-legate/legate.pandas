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

import inspect
import os
from math import log10

import pandas
from pandas.api.types import is_categorical_dtype, is_scalar

from legate.pandas.common import errors as err, types as ty, util as util
from legate.pandas.config import OpCode

from .column import Column, _create_column
from .drop_duplicates import drop_duplicates
from .future import Scalar
from .groupby import GroupbyReducer
from .index import BaseIndex, create_index_from_columns, create_range_index
from .merge import Merger
from .partitioning import HashPartitioner
from .pattern import Map, ScalarMap
from .query import QueryExecutor
from .sort import Sorter

_REVERSED_OPS = set(
    ["radd", "rfloordiv", "rmod", "rmul", "rpow", "rsub", "rtruediv"]
)


class Table(object):
    """
    A Legate implementation of DataFrame.

    The data in a single Table is stored in one or more
    Column objects. When multiple Column objects
    are used, they must internally use the same index space.
    Multiple Table objects can share the same Column
    object.
    """

    def __init__(self, runtime, index=None, columns=None):
        assert isinstance(index, BaseIndex)
        self._runtime = runtime
        self._index = index
        self._columns = []
        if columns is not None:
            self._columns = columns.copy()
        # Lists of columns by which the dataframe is partitioned
        self._partition_keys = []

    def _validate_set_axis(self, new_labels, old_labels):
        """Validates the index or columns replacement against the old labels.

        Args:
            new_labels: The labels to replace with.
            old_labels: The labels to replace.

        Returns:
            The validated labels.
        """
        if not isinstance(new_labels, pandas.Index):
            from pandas.core.index import ensure_index

            new_labels = ensure_index(new_labels)
        old_len = len(old_labels)
        new_len = len(new_labels)
        if old_len != new_len:
            raise ValueError(
                "Length mismatch: Expected axis has %d elements, "
                "new values have %d elements" % (old_len, new_len)
            )
        return new_labels

    def __get_index(self):
        return self._index.to_pandas()

    def __set_index(self, index):
        if self._index is None:
            if not isinstance(index, list):
                from pandas.core.index import ensure_index

                index = ensure_index(index)
            self._index = index
        else:
            index = self._validate_set_axis(index, self._index)
            self._index = index
        if not isinstance(self._index, BaseIndex):
            storage = self._runtime.create_storage(len(self._index))
            self._index = self._runtime.create_index_from_pandas(
                storage, self._index
            )

    index = property(__get_index, __set_index)

    def set_index(self, levels, names):
        index = create_index_from_columns(levels, self._index.volume, names)
        return self.replace_columns(self._columns, index=index)

    def reset_index(self, levels, drop):
        assert isinstance(levels, list)
        if drop:
            index = create_range_index(self._index.storage, self._index.volume)
            return Table(self._runtime, index, self._columns)
        else:
            index_columns = self._index._get_levels(levels)
            if len(levels) == self._index.nlevels:
                index = create_range_index(
                    self._index.storage, self._index.volume
                )
            else:
                index = self._index.droplevel(levels)
            return Table(self._runtime, index, index_columns + self._columns)

    def droplevel(self, level):
        return self.replace_columns(index=self._index.droplevel(level))

    def update_legate_index(self, new_index):
        result = self._index.volume.equal_size(new_index.volume)
        if not result.get_scalar().value:
            old_len = self._index.volume.get_value()
            new_len = new_index.volume.get_value()
            raise ValueError(
                f"Length mismatch: Expected axis has {old_len} elements, "
                f"new values have {new_len} elements"
            )

        # TODO: Here we also need to repartition the dataframe to make it
        #       aligned with the new index
        return Table(self._runtime, new_index, self._columns)

    def _to_index(self, name):
        assert len(self._columns) == 1

        return create_index_from_columns(
            self._columns, self._index.volume, [name]
        )

    def update_columns(self, indexer, value):
        value = self._index._align_partition(value)
        for src_idx, tgt_idx in enumerate(indexer):
            self._columns[tgt_idx] = value._columns[src_idx]

    def slice_columns(self, indexer):
        # We access the internal index directly to avoid materializing
        # it to a pandas' index
        columns = []
        chosen = []

        for col_idx in indexer:
            if col_idx < 0:
                continue
            columns.append(self._columns[col_idx])
            chosen.append(col_idx)

        df = Table(self._runtime, self._index, columns)
        df.set_partition_keys(self.transform_partition_keys(chosen))

        return df

    def select_columns(self, indexer):
        if indexer is None:
            return self._columns
        return [
            (self._columns[col_idx] if col_idx >= 0 else None)
            for col_idx in indexer
        ]

    def align_columns(self, indexer, fill_value=None, fill_dtypes=None):
        columns = []
        for idx, col_idx in enumerate(indexer):
            if col_idx >= 0:
                columns.append(self._columns[col_idx])
            else:
                # TODO: Here we materialize the fill value into a column for
                #       simplicity of the logic later on, but this is certainly
                #       wasting memory space. We could alternatively have
                #       a column type that defers the materialization and keeps
                #       the broadcasted scalar value instead.
                dtype = fill_dtypes[idx]
                column = _create_column(
                    self._index.storage,
                    dtype,
                    self._index.primary_ipart,
                    fill_value is None,
                )
                column.fill(fill_value, self._index.volume)
                columns.append(column)
        return Table(self._runtime, self._index, columns)

    def num_columns(self):
        return len(self._columns)

    def get_dtype(self, idx):
        return self._columns[idx].dtype

    def replace_columns(self, columns=None, index=None):
        columns = self._columns if columns is None else columns
        index = self._index if index is None else index
        new_self = Table(self._runtime, index, columns)
        new_self.set_partition_keys(self.partition_keys)
        return new_self

    @property
    def lg_thunk(self):
        assert len(self._columns) == 1
        return self._columns[0]

    @property
    def partition_keys(self):
        return self._partition_keys

    def set_partition_keys(self, keys):
        assert keys is not None
        self._partition_keys = keys

    def is_partitioned_by(self, keys_to_match):
        if len(self._partition_keys) == 0:
            return []
        else:
            try:
                indices = [
                    keys_to_match.index(idx) for idx in self._partition_keys
                ]
                return indices
            except ValueError:
                return []

    def remap_partition_keys(self, mapping):
        return [mapping[k] for k in self._partition_keys]

    def invalidate_partition_keys(self, to_invalidate):
        if any(k in to_invalidate for k in self._partition_keys):
            return []
        else:
            return self._partition_keys

    def transform_partition_keys(self, inv_mapping):
        mapping = {}
        for tgt, src in enumerate(inv_mapping):
            mapping[src] = tgt

        new_keys = [
            mapping[k] if k in mapping else None for k in self._partition_keys
        ]
        if all(k is not None for k in new_keys):
            return new_keys
        else:
            return []

    @staticmethod
    def join_partition_keys(keys1, keys2):
        if set(keys1) == set(keys2):
            return keys1
        else:
            return []

    @property
    def dtypes(self):
        return [col.dtype.to_pandas() for col in self._columns]

    def _get_dtypes(self):
        return [col.dtype for col in self._columns]

    def insert(self, loc, value):
        assert loc >= 0 and loc <= len(self._columns)

        if is_scalar(value):
            value = self.create_column_from_scalar(value)

        assert len(value._columns) == 1
        column = value._columns[0]

        return self.replace_columns(
            self._columns[:loc] + [column] + self._columns[loc:]
        )

    def create_column_from_scalar(self, value):
        assert is_scalar(value)
        assert self._index is not None

        value_dtype = ty.infer_dtype(value)
        column = _create_column(
            self._index.storage, value_dtype, nullable=value is None
        )
        column.fill(value, self._index.volume)

        return Table(self._runtime, self._index, [column])

    def drop_columns(self, to_drop):
        to_drop = set(to_drop)
        columns = []
        to_invalidate = []
        for col_idx, column in enumerate(self._columns):
            if col_idx not in to_drop:
                columns.append(column)
            else:
                to_invalidate.append(col_idx)
        df = Table(self._runtime, self._index, columns)
        df.set_partition_keys(self.invalidate_partition_keys(to_invalidate))
        return df

    def copy(self):
        return self.slice_rows_by_slice(slice(None), False)

    def broadcast(self, num_columns):
        assert len(self._columns) == 1
        return Table(self._runtime, self._index, self._columns * num_columns)

    def merge(
        self,
        right,
        left_column_names,
        right_column_names,
        how="inner",
        on=None,
        left_on=None,
        right_on=None,
        left_index=False,
        right_index=False,
        sort=False,
        suffixes=("_x", "_y"),
        method="hash",
    ):
        merger = Merger(
            self._runtime,
            self,
            right,
            left_column_names,
            right_column_names,
            method,
            how,
            on,
            left_on,
            right_on,
            left_index,
            right_index,
            suffixes,
        )

        (
            result_index,
            result_column_names,
            result_columns,
            partition_keys,
        ) = merger.perform_merge()

        df = Table(self._runtime, result_index, result_columns)
        df.set_partition_keys(partition_keys)

        return df, result_column_names

    def concat(self, axis, others, **kwargs):
        others = util.to_list_if_scalar(others)
        if axis == 1:
            columns = self._columns.copy()
            for other in others:
                columns.extend(other._columns)
            if len(self._columns) == 0:
                return Table(self._runtime, others[0]._index, columns)
            else:
                return Table(
                    self._runtime,
                    self._index,
                    columns,
                )
        else:
            assert axis == 0

            dfs = [self] + others
            num_dfs = len(dfs)
            result_storage = self._runtime.create_output_storage()
            partition_keys = self.partition_keys

            # FIXME: Here we assumed that dataframes have the same
            #        set of columns. When an input dataframe does
            #        not have a column that any of the inputs do,
            #        it is implicitly extended with a column of nulls
            #        while being concatenated.

            index_dtypes = util.to_list_if_scalar(self._index.dtype)
            value_dtypes = util.get_dtypes(self._columns)

            num_levels = len(index_dtypes)
            num_values = len(value_dtypes)

            all_index_columns = []
            all_value_columns = []

            # Hert the access to the internal _column memeber
            # of the self's index is intentional, as we want to
            # avoid materializing the index unnecessarily
            num_pieces = self._index._column.num_pieces
            for df in dfs:
                index_columns = util.to_list_if_scalar(df._index.column)
                all_index_columns.append(
                    [
                        column.repartition(num_pieces)
                        for column in index_columns
                    ]
                )
                all_value_columns.append(
                    [
                        df._columns[i].repartition(num_pieces)
                        for i in range(num_values)
                    ]
                )
                partition_keys = self.join_partition_keys(
                    partition_keys, df.partition_keys
                )

            nullable_index = [
                any(columns[i].nullable for columns in all_index_columns)
                for i in range(num_levels)
            ]
            nullable_value = [
                any(columns[i].nullable for columns in all_value_columns)
                for i in range(num_values)
            ]

            result_index_columns = result_storage.create_columns(
                index_dtypes, nullable=nullable_index
            )
            result_value_columns = result_storage.create_columns(
                value_dtypes, nullable=nullable_value
            )

            plan = Map(self._runtime, OpCode.CONCATENATE)

            plan.add_scalar_arg(num_levels + num_values, ty.uint32)
            for column in result_index_columns:
                column.add_to_plan_output_only(plan)
            for column in result_value_columns:
                column.add_to_plan_output_only(plan)
            plan.add_scalar_arg(num_dfs, ty.uint32)
            for i in range(num_dfs):
                for column in all_index_columns[i]:
                    column.add_to_plan(plan, True)
                for column in all_value_columns[i]:
                    column.add_to_plan(plan, True)

            launch_domain = self._index._column.launch_domain
            counts = plan.execute(launch_domain)

            result_storage = plan.promote_output_storage(result_storage)
            self._runtime.register_external_weighted_partition(
                result_storage.default_ipart, counts
            )
            del plan

            index_names = util.to_list_if_scalar(self._index.name)

            total_count = counts.cast(ty.int64).sum()
            result_index = create_index_from_columns(
                result_index_columns, total_count, index_names
            )

            result = Table(
                self._runtime,
                result_index,
                result_value_columns,
            )
            result.set_partition_keys(partition_keys)
            return result

    def binary_op(self, op, other):
        reverse = False
        if op in _REVERSED_OPS:
            op = op[1:]
            reverse = True

        # Perform binary operation
        rhs1 = self._columns
        if is_scalar(other):
            other = self._runtime.create_scalar(other, ty.infer_dtype(other))
            rhs2 = [other] * len(rhs1)
        else:
            rhs2 = other._columns

        results = []
        for rh1, rh2 in zip(rhs1, rhs2):
            # If the right operand is integer, we convert it to the left
            # operand's dtype
            if isinstance(rh2, Scalar):
                if ty.is_integer_dtype(rh2.dtype):
                    rh2 = rh2.astype(rh1.dtype)
                elif ty.is_categorical_dtype(rh1.dtype):
                    rh2 = rh1.dtype.encode(rh2, unwrap=False, can_fail=True)
                else:
                    common_dtype = ty.find_common_dtype(rh1.dtype, rh2.dtype)
                    rh1 = rh1.astype(common_dtype)
                    rh2 = rh2.astype(common_dtype)

            elif not (
                ty.is_categorical_dtype(rh1.dtype)
                or ty.is_categorical_dtype(rh2.dtype)
            ):
                common_dtype = ty.find_common_dtype(rh1.dtype, rh2.dtype)
                rh1 = rh1.astype(common_dtype)
                rh2 = rh2.astype(common_dtype)

            lh_dtype = ty.get_binop_result_type(op, rh1.dtype, rh2.dtype)

            if ty.is_string_dtype(rh1.dtype) and op in (
                "add",
                "mul",
            ):
                raise err._unsupported_error(
                    f"unsupported operand type(s) for {op}: "
                    f"'{rh1.dtype}' and '{rh2.dtype}'"
                )

            if reverse:
                rh1, rh2 = rh2, rh1

            swapped = False
            if isinstance(rh1, Scalar):
                rh1, rh2 = rh2, rh1
                swapped = True

            results.append(rh1.binary_op(op, rh2, lh_dtype, swapped=swapped))

        return Table(self._runtime, self._index, results)

    def unary_op(self, op):
        return self.replace_columns(
            [col.unary_op(op) for col in self._columns]
        )

    def isna(self):
        return self.replace_columns([col.isna() for col in self._columns])

    def notna(self):
        return self.replace_columns([col.notna() for col in self._columns])

    def fillna(self, values):
        results = []
        for idx, column in enumerate(self._columns):
            if idx not in values or not column.nullable:
                results.append(column)
            else:
                value = self._runtime.create_scalar(values[idx], column.dtype)
                results.append(column.fillna(value))

        return self.replace_columns(results)

    def query(self, column_names, expr, **kwargs):
        if len(self._columns) == 0:
            return self.copy()

        # Inspect the caller's stackframe to resolve external references
        callframe = inspect.currentframe().f_back.f_back
        callenv = {
            "locals": callframe.f_locals,
            "globals": callframe.f_globals,
        }

        # Allocate a column for the resulting boolean mask
        executor = QueryExecutor(
            self._runtime, expr, callenv, column_names, self
        )
        mask = executor.execute()
        return self.select(mask)

    def sort_index(
        self, axis, levels, ascending, kind, na_position, ignore_index
    ):
        assert axis == 0

        sorter = Sorter(
            self._runtime,
            self,
            None,
            levels,
            ascending,
            na_position,
            ignore_index,
        )

        (index_columns, value_columns) = sorter.sort_values()

        if ignore_index:
            result_index = create_range_index(
                value_columns[0].storage, self._index.volume
            )
        else:
            result_index = create_index_from_columns(
                index_columns,
                self._index.volume,
                self._index.name,
            )

        return self.replace_columns(value_columns, index=result_index)

    def sort_values(
        self, by, axis, ascending, kind, na_position, ignore_index
    ):
        assert axis == 0

        sorter = Sorter(
            self._runtime, self, by, None, ascending, na_position, ignore_index
        )

        (index_columns, value_columns) = sorter.sort_values()

        if ignore_index:
            result_index = create_range_index(
                value_columns[0].storage, self._index.volume
            )
        else:
            result_index = create_index_from_columns(
                index_columns, self._index.volume, self._index.name
            )

        return self.replace_columns(value_columns, index=result_index)

    def slice_index_by_slice(self, sl, is_loc=True):
        bounds = self._index.find_bounds(sl.start, sl.stop, is_loc)
        sliced = Table(self._runtime, self._index, []).slice_rows_by_slice(
            sl, is_loc=is_loc, bounds=bounds
        )
        return sliced._index, bounds

    def slice_index_by_boolean_mask(self, mask):
        sliced = Table(self._runtime, self._index, []).select(mask)
        # Return the mask here cause it could be repartitioned
        return sliced._index

    def slice_rows_by_slice(self, sl, is_loc=True, bounds=None):
        if bounds is None:
            bounds = self._index.find_bounds(sl.start, sl.stop, is_loc)

        rt = self._runtime
        storage = rt.create_output_storage()

        inputs = self._columns.copy()
        if self._index.materialized:
            inputs += util.to_list_if_scalar(self._index.column)

        outputs = [storage.create_similar_column(input) for input in inputs]

        if len(outputs) > 0:
            plan = Map(rt, OpCode.SLICE_BY_RANGE)

            plan.add_future(bounds)
            plan.add_scalar_arg(len(inputs), ty.uint32)
            plan.add_future(self._index.volume)
            for input, output in zip(inputs, outputs):
                input.add_to_plan(plan, True)
                output.add_to_plan_output_only(plan)

            counts = plan.execute(inputs[0].launch_domain)

            storage = plan.promote_output_storage(storage)
            self._runtime.register_external_weighted_partition(
                storage.default_ipart, counts
            )

            volume = counts.cast(ty.int64).sum()
            if self._index.materialized:
                result_index = create_index_from_columns(
                    outputs[len(self._columns) :], volume, self._index.names
                )
            else:
                result_index = self._index.slice_by_bounds(bounds, storage)

            return self.replace_columns(
                outputs[: len(self._columns)], index=result_index
            )
        else:
            result_index = self._index.slice_by_bounds(bounds)
            return self.replace_columns([], index=result_index)

        return self

    def read_at(self, idx):
        return self.slice_rows_by_slice(slice(idx, idx + 1), False)

    def write_at(self, idx, val):
        assert self.num_columns() == 1
        idx = self._runtime.create_future(idx, ty.int64)
        result_column = self._columns[0].write_at(idx, val)
        return self.replace_columns([result_column])

    def scatter_by_boolean_mask(self, mask, lhs_index, value):
        if isinstance(mask, Table):
            assert mask.num_columns() == 1
            mask = mask._columns[0]

        value_is_scalar = is_scalar(value)
        if value_is_scalar:
            inputs = [
                self._runtime.create_scalar(value, target.dtype)
                for target in self._columns
            ]

        else:
            # TODO: For now we assume that lhs_index and value are partitioned
            #       in the same way, which doesn't always hold.
            inputs = value._columns

        output_storage = self._columns[0].storage
        outputs = [
            output_storage.create_similar_column(target)
            for target in self._columns
        ]

        plan = Map(self._runtime, OpCode.SCATTER_BY_MASK)
        plan.add_scalar_arg(value_is_scalar, ty.bool)
        mask.add_to_plan(plan, True)
        plan.add_scalar_arg(self.num_columns(), ty.uint32)
        for (output, target, input) in zip(outputs, self._columns, inputs):
            output.add_to_plan_output_only(plan)
            target.add_to_plan(plan, True)
            input.add_to_plan(plan, True)
        plan.execute(mask.launch_domain)

        return self.replace_columns(outputs)

    def scatter_by_slice(self, lhs_index, bounds, value):
        value_is_scalar = is_scalar(value)

        if value_is_scalar:
            inputs = [
                self._runtime.create_scalar(value, target.dtype)
                for target in self._columns
            ]

        else:
            # TODO: For now we assume that lhs_index and value are partitioned
            #       in the same way, which doesn't always hold.
            inputs = value._columns

        output_storage = self._columns[0].storage
        outputs = [
            output_storage.create_similar_column(input)
            for input in self._columns
        ]

        plan = Map(self._runtime, OpCode.SCATTER_BY_SLICE)
        plan.add_scalar_arg(value_is_scalar, ty.bool)
        plan.add_future(bounds)
        plan.add_scalar_arg(self.num_columns(), ty.uint32)
        for (output, target, input) in zip(outputs, self._columns, inputs):
            output.add_to_plan_output_only(plan)
            target.add_to_plan(plan, True)
            input.add_to_plan(plan, True)
        plan.execute(self._columns[0].launch_domain)

        return self.replace_columns(outputs)

    # FIXME: head and tail in the vanilla Pandas return views of the dataframe,
    #        implying that all changes to the original dataframe are visible
    #        in these views. To keep our storage implementation simple,
    #        however, we will make a copy of the portion for now.
    def head(self, n):
        return self.slice_rows_by_slice(slice(None, n), is_loc=False)

    def tail(self, n):
        return self.slice_rows_by_slice(slice(-n, None), is_loc=False)

    def front(self, n):
        return self.replace_columns(self._columns[:n])

    def back(self, n):
        return self.replace_columns(self._columns[-n:])

    def astype(self, dtypes):
        result_columns = [
            column.to_category_column(dtype)
            if is_categorical_dtype(dtype)
            else column.astype(ty.to_legate_dtype(dtype))
            for column, dtype in zip(self._columns, dtypes)
        ]
        return self.replace_columns(result_columns)

    def unary_reduction(
        self,
        op,
        columns,
        axis=0,
        skipna=True,
        level=None,
    ):
        results = [column.unary_reduction(op) for column in self._columns]

        result_dtypes = [
            ty.get_reduction_result_type(op, col.dtype)
            for col in self._columns
        ]

        if len(set(result_dtypes)) == 1:
            storage = self._runtime.create_storage(len(columns))
            volume = self._runtime.create_future(len(columns), ty.int64)
            index_column = self._runtime._create_string_column_from_pandas(
                storage,
                pandas.Series(columns, dtype=pandas.StringDtype()),
                num_pieces=1,
            )
            result_index = create_index_from_columns([index_column], volume)

            result_column = _create_column(
                storage, result_dtypes[0], index_column.primary_ipart, True
            )

            plan = Map(self._runtime, OpCode.TO_COLUMN)
            result_column.add_to_plan_output_only(plan)
            for result in results:
                plan.add_future(result)
            plan.execute(result_column.launch_domain)

            return Table(self._runtime, result_index, [result_column])

        else:
            err._warning(
                "Series with mixed type values are not supported yet. "
                "The result will be transposed."
            )
            storage = self._runtime.create_storage(1)
            volume = self._runtime.create_future(1, ty.int64)
            index_column = self._runtime._create_string_column_from_pandas(
                storage,
                pandas.Series([op], dtype=pandas.StringDtype()),
                num_pieces=1,
            )
            result_index = create_index_from_columns([index_column], volume)

            result_columns = [
                _create_column(
                    storage, result_dtype, index_column.primary_ipart, True
                )
                for result_dtype in result_dtypes
            ]

            for column, scalar in zip(result_columns, results):
                plan = Map(self._runtime, OpCode.TO_COLUMN)
                column.add_to_plan_output_only(plan)
                plan.add_future(scalar)
                plan.execute(column.launch_domain)

            return Table(self._runtime, result_index, result_columns)

    def copy_if_else(self, cond, other=None, negate=False):
        assert len(self._columns) == len(cond._columns)
        assert is_scalar(other) or len(self._columns) == len(other._columns)
        results = []
        for idx, input in enumerate(self._columns):
            cnd = cond._columns[idx]
            oth = other if is_scalar(other) else other._columns[idx]
            results.append(input.copy_if_else(cnd, oth, negate))
        return self.replace_columns(results)

    def scan_op(self, op, axis=None, skipna=True):
        assert axis == 0

        results = [column.scan_op(op, skipna) for column in self._columns]

        return self.replace_columns(results)

    def drop_duplicates(self, subset, keep, ignore_index):
        inputs = self._columns.copy()
        if not ignore_index:
            inputs += self._index.columns

        (outputs, storage, volume) = drop_duplicates(
            self._runtime,
            inputs,
            subset,
            keep,
        )

        if ignore_index:
            result_index = create_range_index(storage, volume)
        else:
            result_index = create_index_from_columns(
                outputs[len(self._columns) :], volume, self._index.names
            )
            outputs = outputs[: len(self._columns)]

        return self.replace_columns(outputs, index=result_index)

    def dropna(self, axis, idxr, thresh):
        assert axis == 0
        assert idxr is not None

        result_storage = self._runtime.create_output_storage()

        result_columns = []
        result_index_columns = []

        plan = Map(self._runtime, OpCode.DROPNA)

        plan.add_scalar_arg(thresh, ty.uint32)

        plan.add_scalar_arg(len(idxr), ty.uint32)
        for idx in idxr:
            plan.add_scalar_arg(idx, ty.int32)

        num_columns = len(self._columns)
        plan.add_scalar_arg(num_columns, ty.uint32)
        for i in range(num_columns):
            input = self._columns[i]

            output = result_storage.create_similar_column(input)
            result_columns.append(output)

            input.add_to_plan(plan, True)
            output.add_to_plan_output_only(plan)

        index_dtypes = util.to_list_if_scalar(self._index.dtype)
        plan.add_scalar_arg(len(index_dtypes), ty.uint32)

        input_index_materialized = self._index.materialized
        plan.add_scalar_arg(input_index_materialized, ty.bool)

        if input_index_materialized:
            input_index_columns = util.to_list_if_scalar(self._index.column)
            for input, index_dtype in zip(input_index_columns, index_dtypes):
                output = result_storage.create_column(
                    index_dtype, nullable=input.nullable
                )
                result_index_columns.append(output)

                input.add_to_plan(plan, True)
                output.add_to_plan_output_only(plan)
        else:
            plan.add_future(self._index._start)
            plan.add_future(self._index._step)
            for index_dtype in index_dtypes:
                output = result_storage.create_column(
                    index_dtype, nullable=False
                )
                output.add_to_plan_output_only(plan)
                result_index_columns.append(output)

        counts = plan.execute(self._columns[0].launch_domain)
        volume = counts.cast(ty.int64).sum()

        result_storage = plan.promote_output_storage(result_storage)
        self._runtime.register_external_weighted_partition(
            result_storage.default_ipart, counts
        )
        del plan

        result_index = create_index_from_columns(
            result_index_columns, volume, self._index.names
        )

        return self.replace_columns(result_columns, index=result_index)

    def equals(self, other, unwrap=True):
        if any(
            c1.dtype != c2.dtype
            for c1, c2 in zip(self._columns, other._columns)
        ):
            return False

        other = self._index._align_partition(other)

        results = [self._index.equals(other._index, False)]
        for lhs, rhs in zip(self._columns, other._columns):
            results.append(lhs.equals(rhs, False))

        result = self._runtime.all(results)

        if unwrap:
            result = result.get_scalar().value

        return result

    def groupby_reduce(self, key_indices, ops, method, sort):
        groupby = GroupbyReducer(self._runtime, self, key_indices, ops, method)

        (
            total_count,
            keys,
            values,
            partition_keys,
        ) = groupby.perform_groupby()

        result_index = create_range_index(keys[0].storage, total_count)
        df = Table(self._runtime, result_index, keys + values)
        df.set_partition_keys(partition_keys)
        if sort:
            num_keys = len(keys)
            df = df.sort_values(
                list(range(num_keys)), 0, [True] * num_keys, "", "last", True
            )

        return df

    def _shuffle(self, key_indices):
        partitioner = HashPartitioner(self._runtime)

        num_columns = len(self._columns)
        inputs = self._columns.copy()
        if self._index.materialized:
            inputs.extend(util.to_list_if_scalar(self._index.column))
        outputs = partitioner._hash_partition(inputs, key_indices)
        if self._index.materialized:
            result_index = create_index_from_columns(
                outputs[num_columns:],
                self._index.volume,
                util.to_list_if_not_none(self._index.name),
            )
            outputs = outputs[:num_columns]
        else:
            result_index = create_range_index(
                outputs[0].storage, self._index.volume
            )

        result = self.replace_columns(outputs, index=result_index)
        result.set_partition_keys(key_indices)
        return result

    def _is_series(self):
        return self._columns[0] == "__reduced__" and len(self._columns) == 1

    def select(self, mask):
        if isinstance(mask, Table):
            assert len(mask._columns) == 1
            mask = mask._columns[0]

        if self._runtime.debug:
            assert isinstance(mask, Column)
            assert mask.dtype == ty.bool

        result_storage = self._runtime.create_output_storage()

        result_columns = []
        result_index_columns = []

        plan_compact = Map(self._runtime, OpCode.COMPACT)

        mask.add_to_plan(plan_compact, True)

        num_columns = len(self._columns)
        plan_compact.add_scalar_arg(num_columns, ty.uint32)
        for i in range(num_columns):
            input = self._columns[i]

            output = result_storage.create_similar_column(input)
            result_columns.append(output)

            input.add_to_plan(plan_compact, True)
            output.add_to_plan_output_only(plan_compact)

        index_dtypes = util.to_list_if_scalar(self._index.dtype)
        plan_compact.add_scalar_arg(len(index_dtypes), ty.uint32)

        input_index_materialized = self._index.materialized
        plan_compact.add_scalar_arg(input_index_materialized, ty.bool)

        if input_index_materialized:
            input_index_columns = util.to_list_if_scalar(self._index.column)
            for input, index_dtype in zip(input_index_columns, index_dtypes):
                output = result_storage.create_column(
                    index_dtype, nullable=input.nullable
                )
                result_index_columns.append(output)

                input.add_to_plan(plan_compact, True)
                output.add_to_plan_output_only(plan_compact)
        else:
            plan_compact.add_future(self._index._start)
            plan_compact.add_future(self._index._step)
            for index_dtype in index_dtypes:
                output = result_storage.create_column(
                    index_dtype, nullable=False
                )
                output.add_to_plan_output_only(plan_compact)
                result_index_columns.append(output)

        counts = plan_compact.execute(mask.launch_domain)
        volume = counts.cast(ty.int64).sum()

        result_storage = plan_compact.promote_output_storage(result_storage)
        self._runtime.register_external_weighted_partition(
            result_storage.default_ipart, counts
        )
        del plan_compact

        result_index = create_index_from_columns(
            result_index_columns, volume, self._index.names
        )

        return self.replace_columns(result_columns, index=result_index)

    def to_pandas(self, schema_only=False, **kwargs):
        # TODO: Need to cache the dataframe
        all_pandas_series = {}
        for col_idx, col in enumerate(self._columns):
            all_pandas_series[col_idx] = col.to_pandas(schema_only)
        df = pandas.DataFrame(all_pandas_series)

        if self._index is not None:
            index = self._index.to_pandas(schema_only)
            df.index = index

        return df

    def to_numpy(self, **kwargs):
        assert len(self._columns) == 1
        return self._columns[0].to_numpy()

    def _create_directory(self, path):
        # TODO: This better be done in Python
        plan = ScalarMap(self._runtime, OpCode.CREATE_DIR, ty.int32)
        plan.add_scalar_arg(path, ty.string)
        token = plan.execute_single()

        err_code = token.get_value()
        if err_code != 0:
            raise FileNotFoundError(
                f"[Errno {err_code}] No such file or directory: '{path}']"
            )

        return token

    def to_csv(
        self,
        path=None,
        sep=",",
        na_rep="",
        columns=None,
        header=True,
        index=True,
        line_terminator=None,
        chunksize=None,
        partition=False,
        column_names=None,
    ):
        columns = self._columns.copy()

        if index:
            columns = util.to_list_if_scalar(self._index.column) + columns
            column_names = (
                util.to_list_if_scalar(self._index.name) + column_names
            )
            column_names = [
                na_rep if name is None else name for name in column_names
            ]

        if not partition:
            columns = [column.repartition(1) for column in columns]

        plan = Map(self._runtime, OpCode.TO_CSV)
        num_pieces = columns[0].num_pieces

        plan.add_scalar_arg(num_pieces, ty.uint32)
        plan.add_scalar_arg(chunksize, ty.uint32)
        plan.add_scalar_arg(partition, ty.bool)
        plan.add_scalar_arg(header, ty.bool)
        plan.add_scalar_arg(path, ty.string)
        plan.add_scalar_arg(sep, ty.string)
        plan.add_scalar_arg(na_rep, ty.string)
        plan.add_scalar_arg(line_terminator, ty.string)
        plan.add_scalar_arg(len(columns), ty.uint32)
        for column_name in column_names:
            plan.add_scalar_arg(column_name, ty.string)
        for column in columns:
            column.add_to_plan(plan, True)

        fm = plan.execute(columns[0].launch_domain)
        # Since we don't have a dependence mechanism to chain up tasks based on
        # their IO requirements, we need to block on these IO tasks so that
        # the effects are visible to the user upon the return of this function.
        fm.wait()

    def to_parquet(
        self,
        path,
        column_names,
        engine="auto",
        compression="snappy",
        index=None,
        partition_cols=None,
        **kwargs,
    ):
        token = self._create_directory(path)

        def _generate_pandas_metadata(
            table, column_names, index, materialized
        ):
            pandas_schema = table.to_pandas(schema_only=True)
            pandas_schema.columns = column_names

            index_descs = []
            if index is not False:
                if index is None and not materialized:
                    index_descs = [
                        {
                            "kind": "range",
                            "name": table._index.name,
                            "start": table._index.start,
                            "stop": table._index.stop,
                            "step": table._index.step,
                        }
                    ]
                else:
                    index_descs = [
                        f"__index_level_{level}__" if name is None else name
                        for level, name in enumerate(
                            util.to_list_if_scalar(table._index.name)
                        )
                    ]
                    column_names = index_descs + column_names

            if isinstance(pandas_schema.index, pandas.MultiIndex):
                index_levels = pandas_schema.index.levels
            else:
                index_levels = util.to_list_if_scalar(pandas_schema.index)

            from pyarrow import pandas_compat

            metadata = pandas_compat.construct_metadata(
                pandas_schema,
                column_names,
                index_levels,
                index_descs,
                index is not False,
                [col.dtype.to_arrow() for col in table._columns],
            )
            return metadata[str.encode("pandas")].decode(), index_descs

        materialized = self._index.materialized
        metadata, index_descs = _generate_pandas_metadata(
            self, column_names, index, materialized
        )

        columns = self._columns
        if index or (index is not False and materialized):
            columns = util.to_list_if_scalar(self._index.column) + columns
            column_names = index_descs + column_names
        assert len(columns) == len(column_names)

        compression = self._runtime.get_compression_type(compression)

        plan = Map(self._runtime, OpCode.TO_PARQUET)
        num_pieces = self._columns[0].num_pieces

        plan.add_future(token)
        plan.add_scalar_arg(num_pieces, ty.uint32)
        plan.add_scalar_arg(compression, ty.uint32)
        plan.add_scalar_arg(path, ty.string)
        plan.add_scalar_arg(metadata, ty.string)
        plan.add_scalar_arg(len(columns), ty.uint32)
        for column_name in column_names:
            plan.add_scalar_arg(column_name, ty.string)
        for column in columns:
            column.add_to_plan(plan, True)

        fm = plan.execute(self._columns[0].launch_domain)
        # TODO: Once we move the metadata generation to a Python task,
        #       we can avoid blocking here and instead chain the task
        #       to it.
        fm.wait()

        # TODO: We wlil move this post processing to a Python task and
        #       get rid of the use of shard id here.
        if self._runtime._this_is_first_node():
            import pyarrow.parquet as pq

            metadata = None
            num_digits = int(log10(num_pieces)) + 1
            for idx in range(num_pieces):
                part = f"part%0{num_digits}d.parquet" % idx
                md = pq.ParquetFile(os.path.sep.join([path, part])).metadata
                md.set_file_path(part)
                if metadata is None:
                    metadata = md
                else:
                    metadata.append_row_groups(md)
            metadata.write_metadata_file(os.path.sep.join([path, "_metadata"]))

    def to_legate_data(self):
        import pyarrow as pa

        data = dict()
        for (idx, column) in enumerate(self._columns):
            field = pa.field(
                f"column{idx}", column.dtype.to_arrow(), column.nullable
            )
            data[field] = column
        return data

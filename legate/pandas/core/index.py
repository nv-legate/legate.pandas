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

import numpy as np
import pandas
from pandas.api.types import is_integer, is_list_like, is_scalar

from legate.pandas.common import types as ty, util as util
from legate.pandas.config import OpCode

from .column import Column
from .future import PandasFuture
from .pattern import Projection, ScalarMap


class BaseIndex(object):
    def _get_level_number(self, level):
        names = self.names
        count = names.count(level)
        if count > 1:
            raise ValueError(
                f"The name {level} occurs multiple times, use a level number"
            )
        try:
            level = self.names.index(level)
        except ValueError as e:
            if not is_integer(level):
                raise KeyError(f"Level {level} not found") from e
            elif level < 0:
                level += self.nlevels
                if level < 0:
                    orig_level = level - self.nlevels
                    raise IndexError(
                        f"Too many levels: Index has only {self.nlevels} "
                        f"levels, {orig_level} is not a valid level number"
                    ) from e
            elif level >= self.nlevels:
                raise IndexError(
                    f"Too many levels: Index has only {self.nlevels} levels, "
                    f"not {level}"
                ) from e
        return level

    def _get_levels(self, levels):
        columns = self.columns
        return [columns[level] for level in levels]

    def _get_level_names(self, levels):
        names = util.to_list_if_scalar(self.name)
        names = [names[lvl] for lvl in levels]

        if any(name is None for name in names):
            if len(names) == 1:
                names[0] = "index"
            else:
                names = [
                    f"level_{lvl}" if name is None else name
                    for lvl, name in zip(levels, names)
                ]
        return names

    def _to_bounds(self, start, stop):
        plan = ScalarMap(self.runtime, OpCode.TO_BOUNDS, ty.range64)

        plan.add_future(self.volume)
        plan.add_future(
            self.runtime.create_scalar(start, ty.int64).get_future()
        )
        plan.add_future(
            self.runtime.create_scalar(stop, ty.int64).get_future()
        )

        return plan.execute_single()

    def _align_partition(self, to_align):
        ipart = self.primary_ipart
        if ipart == to_align._index.primary_ipart:
            return to_align

        new_ipart = self.runtime.create_isomorphic_partition(
            to_align._index.ispace, ipart
        )

        new_index = to_align._index._repartition_by_ipart(new_ipart)
        columns = [
            column.repartition_by_ipart(new_ipart)
            for column in to_align._columns
        ]

        return to_align.replace_columns(columns, index=new_index)

    @property
    def __constructor__(self):
        return type(self)

    def __len__(self):
        raise NotImplementedError()

    def equals(self, other, unwrap=True):
        raise NotImplementedError()

    @property
    def ispace(self):
        raise NotImplementedError()

    @property
    def volume(self):
        raise NotImplementedError()

    @property
    def name(self):
        raise NotImplementedError()

    @property
    def names(self):
        raise NotImplementedError()

    @property
    def dtype(self):
        raise NotImplementedError()

    @property
    def column(self):
        raise NotImplementedError()

    @property
    def materialized(self):
        raise NotImplementedError()

    @property
    def nlevels(self):
        raise NotImplementedError()

    def rename(self, name):
        raise NotImplementedError()

    def copy(self):
        raise NotImplementedError()

    def to_pandas(self):
        raise NotImplementedError()

    def droplevel(self, level):
        levels = util.to_list_if_scalar(level)
        levels = [self._get_level_number(lvl) for lvl in levels]
        if len(levels) >= self.nlevels:
            raise ValueError(
                f"Cannot remove {len(levels)} levels from an index with "
                f"{self.nlevels} levels: at least one level must be left."
            )
        return self._droplevel(levels)

    def _droplevel(self, levels):
        # This should be a no-op, unless the index is a MultiIndex
        return self

    def __iter__(self):
        raise NotImplementedError()

    def __eq__(self, other):
        return self._binary_op("eq", other)

    def __ne__(self, other):
        return self._binary_op("ne", other)


class RangeIndexIterator(object):
    def __init__(self, base):
        self._base = base
        self._index = 0

    def __next__(self):
        # TODO: Implement this function
        raise StopIteration


class RangeIndex(BaseIndex):
    def __init__(
        self,
        column,
        start=None,
        stop=None,
        step=None,
        name=None,
        volume=None,
        materialized=False,
    ):
        assert isinstance(column, Column)
        assert column.dtype == ty.int64
        assert isinstance(start, PandasFuture)
        assert start.dtype == ty.int64
        assert stop.dtype == ty.int64
        assert step.dtype == ty.int64
        assert volume.dtype == ty.int64

        self._column = column
        self._materialized = materialized
        self._start = start
        self._stop = stop
        self._step = step
        self._name = name
        self._volume = volume

    def __len__(self):
        return int(self._volume.get_value())

    def _identical(self, other):
        return (
            isinstance(other, type(self))
            and self.name == other.name
            and self._column is other._column
        )

    @property
    def runtime(self):
        return self._column.runtime

    def equals(self, other, unwrap=True):
        result = self.runtime.create_scalar(False, ty.bool).get_future()
        if self._identical(other):
            result = self.runtime.create_scalar(True, ty.bool).get_future()

        elif isinstance(other, type(self)):
            result = self.runtime.all(
                [
                    self._start.equal_size(other._start),
                    self._stop.equal_size(other._stop),
                    self._step.equal_size(other._step),
                ]
            )

        elif isinstance(other, StoredIndex):
            result = other.equals(self, False)

        if unwrap:
            result = result.get_scalar().value

        return result

    def _materialize(self):
        if not self.materialized:
            self._column.materialize_indices(self._start, self._step)
            self._materialized = True

    @property
    def storage(self):
        return self._column.storage

    @property
    def ispace(self):
        return self._column.ispace

    @property
    def primary_ipart(self):
        return self._column.primary_ipart

    @property
    def volume(self):
        if self._volume is None:
            if self._start.ready and self._stop.ready and self._step.ready:
                start = self._start.get_value()
                stop = self._stop.get_value()
                step = self._step.get_value()
                volume = (stop - 1 - start) // step + 1
                self._volume = self.runtime.create_future(volume, ty.int64)
            else:
                plan = ScalarMap(
                    self.runtime, OpCode.COMPUTE_RANGE_VOLUME, ty.int64
                )
                plan.add_future(self._start)
                plan.add_future(self._stop)
                plan.add_future(self._step)
                self._volume = plan.execute_single()

        return self._volume

    @property
    def start(self):
        return int(self._start.get_value())

    @property
    def stop(self):
        return int(self._stop.get_value())

    @property
    def step(self):
        return int(self._step.get_value())

    def _get_name(self):
        return self._name

    def _set_name(self, name):
        self._name = name

    name = property(_get_name, _set_name)

    def _get_names(self):
        return [self.name]

    def _set_names(self, names):
        self._name = names[0]

    names = property(_get_names, _set_names)

    @property
    def dtype(self):
        return self._column.dtype

    @property
    def column(self):
        self._materialize()
        return self._column

    @property
    def columns(self):
        return [self.column]

    @property
    def materialized(self):
        return False

    @property
    def nlevels(self):
        return 1

    def rename(self, name):
        return RangeIndex(
            self._column,
            start=self._start,
            stop=self._stop,
            step=self._step,
            name=self._name,
            materialized=self._materialized,
        )

    def copy(self):
        return self.rename(None)

    def to_pandas(self, schema_only=False):
        if schema_only:
            return pandas.RangeIndex(0, name=self._name)
        else:
            return pandas.RangeIndex(
                start=self.start,
                stop=self.stop,
                step=self.step,
                name=self._name,
            )

    def __iter__(self):
        return RangeIndexIterator(self)

    def _add_to_plan(self, plan):
        plan.add_future(self._start)
        plan.add_future(self._stop)
        plan.add_future(self._step)

    @classmethod
    def from_pandas(cls, runtime, column, pandas_index):
        start = runtime.create_future(pandas_index.start, ty.int64)
        stop = runtime.create_future(pandas_index.stop, ty.int64)
        step = runtime.create_future(pandas_index.step, ty.int64)
        volume = runtime.create_future(len(pandas_index), ty.int64)
        return cls(
            column,
            start=start,
            stop=stop,
            step=step,
            name=pandas_index.name,
            volume=volume,
        )

    def find_bounds(self, start, stop, is_loc=True):
        if is_loc:
            plan = ScalarMap(
                self.runtime, OpCode.FIND_BOUNDS_IN_RANGE, ty.range64
            )

            self._add_to_plan(plan)
            plan.add_future(
                self.runtime.create_scalar(start, self.dtype).get_future()
            )
            plan.add_future(
                self.runtime.create_scalar(stop, self.dtype).get_future()
            )

            return plan.execute_single()

        else:
            return self._to_bounds(start, stop)

    def to_stored_index(self):
        return StoredIndex(self.column, self._volume, self.name)

    def slice_by_bounds(self, bounds, storage=None):
        plan = ScalarMap(self.runtime, OpCode.COMPUTE_RANGE_START, ty.int64)
        self._add_to_plan(plan)
        plan.add_future(bounds)
        new_start = plan.execute_single()

        plan = ScalarMap(self.runtime, OpCode.COMPUTE_RANGE_STOP, ty.int64)
        self._add_to_plan(plan)
        plan.add_future(bounds)
        new_stop = plan.execute_single()

        plan = ScalarMap(self.runtime, OpCode.COMPUTE_RANGE_VOLUME, ty.int64)
        plan.add_future(new_start)
        plan.add_future(new_stop)
        plan.add_future(self._step)
        new_volume = plan.execute_single()

        # If no storage was passed, we need to make a fresh storage using
        # the new volume and partition it with the subrange sizes so that
        # it aligns with the slicer.
        if storage is None:
            ispace = self.runtime.find_or_create_index_space(new_volume)

            plan = ScalarMap(
                self.runtime, OpCode.COMPUTE_SUBRANGE_SIZES, ty.int64
            )
            # TODO: We want to roll this into the logic in column.add_to_plan
            plan.add_no_access(
                self._column.data, Projection(self._column.primary_ipart, 0)
            )
            plan.add_future(bounds)
            counts = plan.execute(self._column.launch_domain)

            ipart = self.runtime.create_partition_from_weights(
                ispace, self._column.cspace, counts
            )

            storage = self.runtime.create_storage(ispace, ipart)

        return create_range_index(
            storage,
            volume=new_volume,
            name=self._name,
            start=new_start,
            stop=new_stop,
            step=self._step,
        )

    def _binary_op(self, op, other):
        assert is_scalar(other)
        return self.to_stored_index()._binary_op(op, other)

    def _get_drop_mask_for(self, label, _):
        return self != label

    def _repartition_by_ipart(self, new_ipart):
        return RangeIndex(
            self._column.repartition_by_ipart(new_ipart),
            start=self._start,
            stop=self._stop,
            step=self._step,
            name=self._name,
            volume=self._volume,
            materialized=self._materialized,
        )


class StoredIndexIterator(object):
    def __init__(self, base):
        self._base = base
        self._index = 0

    def __next__(self):
        # TODO: Implement this function
        raise StopIteration


class StoredIndex(BaseIndex):
    def __init__(self, column, volume, name=None):
        assert isinstance(column, Column)
        assert isinstance(volume, PandasFuture)
        assert volume.dtype == ty.int64
        self._column = column
        self._volume = volume
        self._name = name

    def __len__(self):
        return int(self._volume.get_value())

    def _identical(self, other):
        return (
            isinstance(other, type(self))
            and self.name == other.name
            and self._column is other._column
        )

    @property
    def runtime(self):
        return self._column.runtime

    def equals(self, other, unwrap=True):
        result = self.runtime.create_scalar(False, ty.bool).get_future()
        if self._identical(other):
            result = self.runtime.create_scalar(True, ty.bool).get_future()

        elif isinstance(other, RangeIndex):
            result = self.equals(other.to_stored_index(), False)

        elif isinstance(other, type(self)) and self.dtype == other.dtype:
            result = self.column.equals(other.column, False)

        if unwrap:
            result = result.get_scalar().value

        return result

    @property
    def storage(self):
        return self._column.storage

    @property
    def ispace(self):
        return self._column.ispace

    @property
    def primary_ipart(self):
        return self._column.primary_ipart

    @property
    def volume(self):
        return self._volume

    def _get_name(self):
        return self._name

    def _set_name(self, name):
        self._name = name

    name = property(_get_name, _set_name)

    def _get_names(self):
        return [self.name]

    def _set_names(self, names):
        self._name = names[0]

    names = property(_get_names, _set_names)

    @property
    def dtype(self):
        return self._column.dtype

    @property
    def column(self):
        return self._column

    @property
    def columns(self):
        return [self.column]

    @property
    def materialized(self):
        return True

    @property
    def nlevels(self):
        return 1

    def rename(self, name):
        return StoredIndex(self._column, self._volume, name)

    def copy(self):
        return self.rename(None)

    def to_pandas(self, schema_only=False):
        return pandas.Index(
            self._column.to_pandas(schema_only), name=self._name
        )

    def __iter__(self):
        return StoredIndexIterator(self)

    @classmethod
    def from_pandas(cls, runtime, column, pandas_index):
        volume = runtime.create_future(len(pandas_index), ty.int64)
        if isinstance(pandas_index, pandas.CategoricalIndex):
            sorted_categories = pandas_index.categories.sort_values()
            column.data.from_numpy(
                pandas.CategoricalIndex(
                    list(pandas_index), categories=sorted_categories
                ).values.codes.astype(np.int64)
            )
        elif ty.is_string_dtype(column.dtype):
            return cls(column, volume, pandas_index.name)
        else:
            column.data.from_numpy(pandas_index.values)
        return cls(column, volume, pandas_index.name)

    def find_bounds(self, start, stop, is_loc=True):
        if is_loc:
            plan = ScalarMap(self.runtime, OpCode.FIND_BOUNDS, ty.range64)

            plan.add_future(
                self.runtime.create_scalar(start, self.dtype).get_future()
            )
            plan.add_future(
                self.runtime.create_scalar(stop, self.dtype).get_future()
            )
            plan.add_future(self._volume)
            self._column.add_to_plan(plan, True)

            return plan.execute(self._column.launch_domain).reduce("union")

        else:
            return self._to_bounds(start, stop)

    def _binary_op(self, op, other):
        assert is_scalar(other)
        column = self.column
        other = self.runtime.create_scalar(other, column.dtype)
        return column.binary_op(op, other, ty.bool)

    def _get_drop_mask_for(self, label, _):
        return self != label

    def _repartition_by_ipart(self, new_ipart):
        return StoredIndex(
            self._column.repartition_by_ipart(new_ipart),
            self._volume,
            self._name,
        )


class MultiIndexIterator(object):
    def __init__(self, base):
        self._base = base
        self._index = 0

    def __next__(self):
        # TODO: Implement this function
        raise StopIteration


class MultiIndex(BaseIndex):
    _SUPPORTED_BINOPS = {"eq", "ne"}
    _MERGE_OPS = {"eq": "mul", "ne": "add"}

    def __init__(self, levels):
        self._levels = levels

    def __len__(self):
        # TODO: Levels should all have the same length
        return len(self._levels[0])

    def _identical(self, other):
        return (
            isinstance(other, type(self))
            and self.name == other.name
            and all(lh is rh for lh, rh in zip(self._levels, other._levels))
        )

    def _droplevel(self, to_drop):
        to_drop = set(to_drop)
        levels = [
            level
            for lvl_num, level in enumerate(self._levels)
            if lvl_num not in to_drop
        ]
        if len(levels) == 1:
            return levels[0]
        else:
            return MultiIndex(levels)

    @property
    def runtime(self):
        return self._levels[0].runtime

    def equals(self, other, unwrap=True):
        result = self.runtime.create_scalar(False, ty.bool).get_future()
        if self._identical(other):
            result = self._column.runtime.create_scalar(
                True, ty.bool
            ).get_future()

        elif isinstance(other, type(self)) and self.nlevels == other.nlevels:
            results = [
                self.equals(other, False)
                for self, other in zip(self._levels, other._levels)
            ]
            result = self.runtime.all(results)

        if unwrap:
            result = result.get_scalar().value

        return result

    @property
    def storage(self):
        return self._levels[0].storage

    @property
    def ispace(self):
        # TODO: Levels should all have the same index space
        return self._levels[0].ispace

    @property
    def primary_ipart(self):
        return self._levels[0].primary_ipart

    @property
    def volume(self):
        return self._levels[0].volume

    def _get_names(self):
        return [level.name for level in self._levels]

    def _set_names(self, names):
        for level, name in zip(self._levels, names):
            level.name = name

    names = property(_get_names, _set_names)

    name = names

    @property
    def dtype(self):
        return [level.dtype for level in self._levels]

    @property
    def column(self):
        return [level.column for level in self._levels]

    columns = column

    @property
    def materialized(self):
        return all([level.materialized for level in self._levels])

    @property
    def nlevels(self):
        return len(self._levels)

    def rename(self, name):
        return MultiIndex(self._levels, name=name)

    def copy(self):
        return self.rename(None)

    def to_pandas(self, schema_only=False):
        levels = [level.to_pandas(schema_only) for level in self._levels]
        return pandas.MultiIndex.from_arrays(levels)

    def __iter__(self):
        return MultiIndexIterator(self)

    @classmethod
    def from_pandas(cls, levels):
        return cls(levels)

    def find_bounds(self, start, stop, is_loc=True):
        if is_loc:
            assert False

        else:
            return self._to_bounds(start, stop)

    def _binary_op(self, op, other):
        assert op in self._SUPPORTED_BINOPS

        if not util.is_tuple(other):
            other = (other,)

        results = [
            self._levels[lvl]._binary_op(op, oth)
            for lvl, oth in enumerate(other)
        ]

        result = results[0]
        if len(results) > 1:
            op = self._MERGE_OPS[op]
            for other in results[1:]:
                result = result.binary_op(op, other, result.dtype)
        return result

    def _get_drop_mask_for(self, label, level):
        if level is not None:
            level = self._get_level_number(level)
            return self._levels[level] != label

        else:
            return self != label

    def _repartition_by_ipart(self, new_ipart):
        levels = [lvl._repartition_by_ipart(new_ipart) for lvl in self._levels]
        return MultiIndex(levels)


def create_range_index(
    storage, volume=None, name=None, start=None, stop=None, step=None
):
    column = storage.create_column(ty.int64, nullable=False)
    if stop is None:
        stop = volume
        if start is None:
            start = storage._runtime.create_future(0, ty.int64)
        if step is None:
            step = storage._runtime.create_future(1, ty.int64)
    else:
        assert start is not None
        assert stop is not None
        assert step is not None
    return RangeIndex(
        column, start=start, stop=stop, step=step, name=name, volume=volume
    )


def create_index_from_columns(columns, volume, names=None):
    num_levels = len(columns)
    assert num_levels > 0
    if not is_list_like(names):
        names = [names] * num_levels

    columns = [
        column.astype(ty.ensure_valid_index_dtype(column.dtype))
        for column in columns
    ]
    result_indices = [
        StoredIndex(column, volume, name)
        for column, name in zip(columns, names)
    ]
    if num_levels == 1:
        return result_indices[0]
    else:
        return MultiIndex(result_indices)


def create_index_from_pandas(runtime, storage, pandas_index):
    if isinstance(pandas_index, pandas.RangeIndex):
        index_column = storage.create_column(
            ty.to_legate_dtype(pandas_index.dtype),
            nullable=False,
        )
        return RangeIndex.from_pandas(runtime, index_column, pandas_index)
    # Note that DatetimeIndex is a subclass of Int64Index
    elif isinstance(
        pandas_index,
        (
            pandas.RangeIndex,
            pandas.Int64Index,
            pandas.UInt64Index,
            pandas.Float64Index,
            pandas.CategoricalIndex,
        ),
    ):
        if isinstance(pandas_index, pandas.CategoricalIndex):
            index_column = runtime._create_category_column_from_pandas(
                storage, pandas_index.to_series()
            )
        else:
            index_column = storage.create_column(
                # FIXME: For now we assume indices are non-nullable
                ty.to_legate_dtype(pandas_index.dtype),
                nullable=False,
            )
        return StoredIndex.from_pandas(runtime, index_column, pandas_index)
    elif isinstance(pandas_index, pandas.MultiIndex):
        levels = [
            create_index_from_pandas(
                runtime, storage, pandas_index.get_level_values(i)
            )
            for i in range(pandas_index.nlevels)
        ]
        return MultiIndex.from_pandas(levels)
    else:
        # FIXME: This part is a bit wonky now. Will need to clean this up.
        if all(isinstance(v, str) for v in pandas_index):
            index_column = runtime._create_column_from_pandas(
                storage, pandas_index
            )
            return StoredIndex.from_pandas(runtime, index_column, pandas_index)
        else:
            raise NotImplementedError(
                f"Importing {type(pandas_index)} is not yet supported"
            )

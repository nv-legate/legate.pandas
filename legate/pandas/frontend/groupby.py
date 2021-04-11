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

from pandas.api.types import is_dict_like

from legate.pandas.common import errors as err, util as util
from legate.pandas.frontend import reduction as reduction


class GroupBy(object):
    def __init__(
        self, df, by, axis, level, as_index, sort, method, is_series_groupby
    ):
        axis = df._get_axis_number(axis)
        if axis not in (0,):
            raise err._unsupported_error("axis", axis)

        if by is None and level is None:
            raise TypeError("You have to supply one of 'by' and 'level'")

        self._df = df
        self._axis = axis
        self._as_index = as_index
        self._sort = sort
        self._method = method
        self._is_series_groupby = is_series_groupby

        if level is not None:
            levels = util.to_list_if_scalar(level)
            self._keys = [
                df._raw_index._get_level_number(lvl) for lvl in levels
            ]

            # Reset the levels chosen as the groupby keys so that they
            # appear in the frame
            self._df = self._df.reset_index(self._keys)

            # The pushed-out index levels are now the first few columns
            # in the frame, so we should change the key indices to pick
            # them correctly as the groupby keys later

            # A technical note: reset_index internally sorts level
            # numbers before it pushes out the corresponding levels
            # to the dataframe. Therefore, we use argsort to compute
            # the positions of the columns that we later pick for indices.
            self._keys = [
                p[0] for p in sorted(enumerate(self._keys), key=lambda p: p[1])
            ]
            self._levels = self._keys

        else:
            if df._is_series:
                raise err._unsupported_error(
                    f"{type(self._df).__name__} only supports level"
                )

            keys = util.to_list_if_scalar(by)
            if all(not isinstance(key, str) for key in keys):
                raise err._unsupported_error(
                    "groupby keys must be column names for now"
                )

            idxr = []
            columns = df._get_columns()
            for key in keys:
                idx = columns.get_indexer_for([key])
                if len(idx) > 1:
                    raise KeyError(f"ambiguous key name {key}")
                if idx[0] == -1:
                    raise KeyError(key)
                idxr.extend(idx)

            self._keys = idxr
            self._levels = []

    def count(self):
        return self._groupby_reduce(ops=[("count", False)])

    def max(self, numeric_only=False, min_count=-1):
        if min_count > 0:
            raise err._unsupported_error("min_count", min_count)
        return self._groupby_reduce(ops=[("max", numeric_only)])

    def mean(self, numeric_only=True):
        if numeric_only not in (
            True,
            None,
        ):
            raise err._unsupported_error("numeric_only", numeric_only)
        return self._groupby_reduce(ops=[("mean", numeric_only)])

    def min(self, numeric_only=False, min_count=-1):
        if min_count > 0:
            raise err._unsupported_error("min_count", min_count)
        return self._groupby_reduce(ops=[("min", numeric_only)])

    def prod(self, numeric_only=True, min_count=0):
        if numeric_only not in (
            True,
            None,
        ):
            raise err._unsupported_error("numeric_only", numeric_only)
        if min_count > 0:
            raise err._unsupported_error("min_count", min_count)
        return self._groupby_reduce(ops=[("prod", numeric_only)])

    def size(self):
        return self._groupby_reduce(ops=[("size", False)])

    def sum(self, numeric_only=True, min_count=0):
        if numeric_only not in (
            True,
            None,
        ):
            raise err._unsupported_error("numeric_only", numeric_only)
        if min_count > 0:
            raise err._unsupported_error("min_count", min_count)
        return self._groupby_reduce(ops=[("sum", numeric_only)])

    def std(self, ddof=1):
        if ddof != 1:
            raise err._unsupported_error("ddof", ddof)
        return self._groupby_reduce(ops=[("std", True)])

    def var(self, ddof=1):
        if ddof != 1:
            raise err._unsupported_error("ddof", ddof)
        return self._groupby_reduce(ops=[("var", True)])

    def agg(self, *args, **kwargs):
        return self.aggregate(*args, **kwargs)

    def aggregate(
        self, func=None, *args, engine=None, engine_kwargs=None, **kwargs
    ):
        ops = reduction.convert_agg_func(func)
        return self._groupby_reduce(ops)

    def _groupby_reduce(self, ops=None):
        columns = self._df._get_columns()
        dtypes = self._df._frame.dtypes

        valid_ops = {}
        valid_columns = []

        size_reduction = len(ops) == 1 and ops[0][0] == "size"
        # If the ops are given as a list, apply them across all the columns
        # with compatible data types
        if isinstance(ops, list):
            key_indices = set(self._keys)
            for col_idx, col in enumerate(columns):
                if col_idx in key_indices:
                    continue

                if reduction.incompatible_ops(ops, dtypes[col_idx]):
                    continue

                valid_ops[col_idx] = [desc[0] for desc in ops]
                valid_columns.append(col)
                # Special case with the size reduction, which produces a single
                # output regardless of the number of input columns
                if size_reduction:
                    break

        # If the ops are passed in a dictionary, it also specifies the input
        # columns on which the aggregation are performed
        else:
            assert is_dict_like(ops)
            for col, descs in ops.items():
                col_idx = columns.get_indexer_for([col])
                if len(col_idx) > 1:
                    raise KeyError(f"ambiguous column name {col}")
                if col_idx[0] == -1:
                    raise KeyError(col)

                if reduction.incompatible_ops(descs, dtypes[col_idx[0]]):
                    continue

                valid_ops[col_idx[0]] = [desc[0] for desc in descs]
                valid_columns.append(col)

        frame = self._df._frame.groupby_reduce(
            self._keys, valid_ops, self._method, self._sort
        )

        # If more than one aggregation is requested for a column,
        # the output column names should use MultiIndex
        multi_aggs = any(len(set(ops)) > 1 for ops in valid_ops.values())

        def _generate_columns(columns, all_ops):
            if multi_aggs:
                from pandas import MultiIndex

                pairs = []
                for idx, ops in all_ops.items():
                    pairs.extend([(columns[idx], op) for op in ops])
                index = MultiIndex.from_tuples(pairs)

                if self._is_series_groupby:
                    index = index.droplevel(0)

                return index

            else:
                from pandas import Index

                return Index([columns[idx] for idx in all_ops.keys()])

        from .dataframe import DataFrame

        if self._as_index:
            # Groupby keys are rearranged to come first in the frame,
            # no matter where they were in the input frame, so the
            # indexer should be picking the first N keys in the frame,
            # where N is the number of keys
            indexer = list(range(len(self._keys)))
            index_columns = frame.select_columns(indexer)

            # However, the index names are derived from the input
            # dataframe, which is not rearranged, so we use the original
            # indexer to select the names
            index_names = columns[self._keys]

            value_names = _generate_columns(columns, valid_ops)

            # Once we find the index columns, we drop them from the frame
            frame = frame.drop_columns(indexer)
            frame = frame.set_index(index_columns, index_names)

            if size_reduction or (self._is_series_groupby and not multi_aggs):
                # Size reduction always produces a series
                from .series import Series

                return Series(frame=frame, name=value_names[0])
            else:
                return DataFrame(frame=frame, columns=value_names)

        else:
            # Index levels don't survive in the output when as_index is False
            levels = set(self._levels)
            keys = [key for key in self._keys if key not in levels]

            key_names = columns[keys]
            value_names = _generate_columns(columns, valid_ops)

            # If the column names are stored in a MultiIndex,
            # we should extend the key names to match the shape
            if multi_aggs:
                from pandas import MultiIndex

                key_names = MultiIndex.from_arrays(
                    [key_names, [""] * len(key_names)]
                )

            value_names = key_names.append(value_names)
            frame = frame.drop_columns(self._levels)
            return DataFrame(frame=frame, columns=value_names)


class DataFrameGroupBy(GroupBy):
    def __init__(
        self,
        df,
        by,
        axis,
        level,
        as_index,
        sort,
        method,
    ):
        super().__init__(df, by, axis, level, as_index, sort, method, False)


class SeriesGroupBy(GroupBy):
    def __init__(
        self,
        df,
        by,
        axis,
        level,
        as_index,
        sort,
        method,
    ):
        super().__init__(df, by, axis, level, as_index, sort, method, True)

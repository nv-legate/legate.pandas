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

import os

import numpy as np
import pandas
from pandas.api.types import (
    is_bool_dtype,
    is_dict_like,
    is_list_like,
    is_numeric_dtype,
    is_scalar,
)
from pandas.util._validators import validate_bool_kwarg

from legate.pandas.common import errors as err, util as util
from legate.pandas.frontend import reduction as reduction

from .doc_utils import copy_docstring

# We ask to these empty Pandas objects when we don't want to or know how to
# handle arguments.
_empty_pandas_objects = {
    "DataFrame": pandas.DataFrame(),
    "Series": pandas.Series(),
}


class Frame(object):
    """
    This class provides a set of functions shared by DataFrame and Series.
    """

    ###########
    # Utilities
    ###########

    @property
    def __name__(self):
        return type(self).__name__

    def __ctor__(self, *args, **kwargs):
        return type(self)(*args, **kwargs)

    @classmethod
    def _get_empty_pandas_object(cls):
        return _empty_pandas_objects[cls.__name__]

    @classmethod
    def _get_axis_number(cls, axis, default=None):
        if axis is None:
            return default
        else:
            return cls._get_empty_pandas_object()._get_axis_number(axis)

    def _get_dtypes(self):
        return self._frame.dtypes

    def _update_frame(self, new_frame):
        self._create_or_update_frame(new_frame, True)

    @property
    def _raw_index(self):
        return self._frame._index

    @property
    def _runtime(self):
        return self._frame._runtime

    def _ensure_valid_frame(self, data, copy=False):
        if is_scalar(data) or util.is_tuple(data):
            return data
        elif isinstance(data, Frame):
            return data.copy(deep=copy)
        elif isinstance(data, pandas.DataFrame):
            from .dataframe import DataFrame

            return DataFrame(data)
        elif isinstance(data, pandas.Series):
            from .series import Series

            return Series(data)
        elif isinstance(data, np.ndarray):
            # TODO: Here we assume that the axis to which we align the ndarray
            #       is the index, but we really should be choosing between
            #       the index and the columns, depending on the axis argument.
            if data.ndim == 1:
                from .series import Series

                if len(self) != len(data):
                    raise ValueError(
                        f"Length of passed values is {len(self)}, "
                        f"index implies {len(data)}."
                    )

                name = self.name if self._is_series else None
                return Series(data, name=name, index=self._raw_index)
            elif data.ndim == 2:
                if self._is_series:
                    raise Exception("Data must be 1-dimensional")

                from .dataframe import DataFrame

                return DataFrame(
                    data, columns=self.columns, index=self._raw_index
                )
            else:
                raise ValueError("array must be either 1-d or 2-d")

        elif is_list_like(data):
            return self._ensure_valid_frame(np.array(data))

        else:
            raise ValueError(f"unsupported value type '{type(data)}'")

    ################################
    # Attributes and underlying data
    ################################

    def _set_index(self, new_index):
        self._frame.index = new_index

    def _get_index(self):
        return self._frame.index

    index = property(
        _get_index, _set_index, None, pandas.DataFrame.index.__doc__
    )

    @copy_docstring(pandas.DataFrame.size)
    @property
    def size(self):
        return len(self) * len(self._get_columns())

    @copy_docstring(pandas.DataFrame.empty)
    @property
    def empty(self):
        return len(self.columns) == 0

    def __len__(self):
        return len(self._raw_index)

    def __str__(self):
        return repr(self)

    ############
    # Conversion
    ############

    @copy_docstring(pandas.DataFrame.astype)
    def astype(self, dtype, copy=True, errors="raise"):
        if errors not in ("raise",):
            raise err._unsupported_error("errors", errors)

        columns = [self.name] if self._is_series else self.columns
        if isinstance(dtype, dict):
            if errors == "raise" and any(
                key in columns for key in dtype.keys()
            ):
                raise KeyError(
                    "Only a column name can be used for the key in a dtype "
                    "mappings argument."
                )
            dtypes = [dtype.get(col, None) for col in columns]
        else:
            dtypes = [dtype] * len(columns)

        new_frame = self._frame.astype(dtypes)
        return self._create_or_update_frame(new_frame, not copy)

    def __copy__(self, deep=True):
        return self.copy(deep=deep)

    def __deepcopy__(self, memo=None):
        return self.copy(deep=True)

    @copy_docstring(pandas.DataFrame.bool)
    def bool(self):
        shape = self.shape
        if shape != (1,) and shape != (1, 1):
            raise ValueError(
                "The PandasObject does not have exactly "
                "1 element. Return the bool of a single "
                "element PandasObject. The truth value is "
                "ambiguous. Use a.empty, a.item(), a.any() "
                "or a.all()."
            )
        else:
            return self._to_pandas().bool()

    def __nonzero__(self):
        raise ValueError(
            "The truth value of a {0} is ambiguous. "
            "Use a.empty, a.bool(), a.item(), a.any() or a.all().".format(
                self.__class__.__name__
            )
        )

    __bool__ = __nonzero__

    #####################
    # Indexing, iteration
    #####################

    def _copy_if_else(
        self,
        cond,
        other=None,
        inplace=False,
        axis=None,
        level=None,
        errors="raise",
        try_cast=False,
        negate=False,
    ):
        inplace = validate_bool_kwarg(inplace, "inplace")
        axis = self._get_axis_number(axis, 0)

        if level is not None:
            raise err._unsupported_error("level", level)

        if axis not in (0,):
            raise err._unsupported_error("axis", axis)

        if try_cast not in (False,):
            raise err._unsupported_error("try_cast", try_cast)

        # Checks on cond
        cond = self._ensure_valid_frame(cond)

        if self.ndim < cond.ndim:
            raise ValueError(
                "cannot use the higher dimensional dataframe for 'cond'"
            )
        _, cond = self._align_frame(cond, join="left", broadcast_axis=1)

        if any(not is_bool_dtype(dtype) for dtype in cond._get_dtypes()):
            raise ValueError("'cond' must have only boolean values")

        # Checks on other
        if not is_scalar(other):
            other = self._ensure_valid_frame(other)

            if self.ndim < other.ndim:
                raise ValueError(
                    "cannot use the higher dimensional dataframe for 'other'"
                )
            _, other = self._align_frame(other, join="left", broadcast_axis=1)

            for l_dtype, r_dtype in zip(
                self._get_dtypes(), other._get_dtypes()
            ):
                if l_dtype != r_dtype:
                    raise ValueError("'other' must have the same type as self")

            other = other._frame

        else:
            other = util.sanitize_scalar(other)

        frame = self._frame.copy_if_else(cond._frame, other, negate=negate)
        return self._create_or_update_frame(frame, inplace)

    @copy_docstring(pandas.DataFrame.head)
    def head(self, n=5):
        return self._create_or_update_frame(self._frame.head(n), False)

    @copy_docstring(pandas.DataFrame.tail)
    def tail(self, n=5):
        return self._create_or_update_frame(self._frame.tail(n), False)

    @copy_docstring(pandas.DataFrame.get)
    def get(self, key, default=None):
        if key in self.keys():
            return self.__getitem__(key)
        else:
            return default

    ###########################
    # Binary operator functions
    ###########################

    def _binary_op(self, op, other, axis=None, level=None, fill_value=None):
        # Retrieve arguments and convert them to default ones if necessary
        axis = self._get_axis_number(axis)

        # Raise an exception for cases that are not implemented yet
        if level is not None:
            raise err._unsupported_error("level", level)

        other = self._ensure_valid_frame(other)

        if not self._is_series and not is_scalar(other):
            if other._is_series and axis not in (0,):
                raise err._unsupported_error("axis", axis)

        # Convert the RHS to a frame unless it's a scalar
        if is_scalar(other):
            new_self = self
            other = util.sanitize_scalar(other)

        else:
            new_self, other = self._align_frame(
                other, join="outer", fill_value=fill_value, broadcast_axis=1
            )
            other = other._frame

        new_frame = new_self._frame.binary_op(op, other)

        if new_self._is_series:
            from .series import Series

            return Series(frame=new_frame, name=new_self.name)
        else:
            from .dataframe import DataFrame

            return DataFrame(frame=new_frame, columns=new_self.columns)

    ########################################
    # Function application, GroupBy & window
    ########################################

    ##################################
    # Computations / descriptive stats
    ##################################

    @copy_docstring(pandas.DataFrame.abs)
    def abs(self):
        new_frame = self._frame.unary_op("abs")
        return self._create_or_update_frame(new_frame, inplace=False)

    def _unary_reduction(self, op, axis=0, skipna=True, level=None, **kwargs):
        return reduction.unary_reduction(
            self, op, axis=axis, skipna=skipna, level=level
        )

    @copy_docstring(pandas.DataFrame.all)
    def all(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        axis = self._get_axis_number(axis) if axis is not None else 0
        if bool_only not in (
            None,
            False,
        ):
            raise err._unsupported_error("bool_only", bool_only)
        return self._unary_reduction(
            [("all", True)],
            axis=axis,
            skipna=skipna,
            level=level,
            bool_only=bool_only,
            **kwargs,
        )

    @copy_docstring(pandas.DataFrame.any)
    def any(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        axis = self._get_axis_number(axis) if axis is not None else 0
        if bool_only not in (
            None,
            False,
        ):
            raise err._unsupported_error("bool_only", bool_only)
        return self._unary_reduction(
            [("any", True)],
            axis=axis,
            skipna=skipna,
            level=level,
            bool_only=bool_only,
            **kwargs,
        )

    @copy_docstring(pandas.DataFrame.count)
    def count(self, axis=0, level=None, numeric_only=False):
        axis = self._get_axis_number(axis) if axis is not None else 0
        if numeric_only not in (
            None,
            False,
        ):
            raise err._unsupported_error("numeric_only", numeric_only)
        return self._unary_reduction(
            [("count", numeric_only)], axis=axis, level=level
        )

    @copy_docstring(pandas.DataFrame.cummax)
    def cummax(self, axis=None, skipna=True, *args, **kwargs):
        axis = self._get_axis_number(axis, 0)
        if axis != 0:
            raise err._unsupported_error("axis", axis)
        return self._create_or_update_frame(
            self._frame.scan_op("cummax", axis=axis, skipna=skipna), False
        )

    @copy_docstring(pandas.DataFrame.cummin)
    def cummin(self, axis=None, skipna=True, *args, **kwargs):
        axis = self._get_axis_number(axis, 0)
        if axis != 0:
            raise err._unsupported_error("axis", axis)
        return self._create_or_update_frame(
            self._frame.scan_op("cummin", axis=axis, skipna=skipna), False
        )

    @copy_docstring(pandas.DataFrame.cumprod)
    def cumprod(self, axis=None, skipna=True, *args, **kwargs):
        axis = self._get_axis_number(axis, 0)
        if axis != 0:
            raise err._unsupported_error("axis", axis)
        return self._create_or_update_frame(
            self._frame.scan_op("cumprod", axis=axis, skipna=skipna), False
        )

    @copy_docstring(pandas.DataFrame.cumsum)
    def cumsum(self, axis=None, skipna=True, *args, **kwargs):
        axis = self._get_axis_number(axis, 0)
        if axis != 0:
            raise err._unsupported_error("axis", axis)
        return self._create_or_update_frame(
            self._frame.scan_op("cumsum", axis=axis, skipna=skipna), False
        )

    @copy_docstring(pandas.DataFrame.max)
    def max(
        self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs
    ):
        axis = self._get_axis_number(axis) if axis is not None else 0
        if numeric_only not in (
            None,
            False,
        ):
            raise err._unsupported_error("numeric_only", numeric_only)
        return self._unary_reduction(
            [("max", numeric_only)],
            axis=axis,
            skipna=skipna,
            level=level,
            **kwargs,
        )

    @copy_docstring(pandas.DataFrame.mean)
    def mean(
        self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs
    ):
        axis = self._get_axis_number(axis) if axis is not None else 0
        if numeric_only not in (
            None,
            True,
        ):
            raise err._unsupported_error("numeric_only", numeric_only)
        return self._unary_reduction(
            [("mean", numeric_only)],
            axis=axis,
            skipna=skipna,
            level=level,
            **kwargs,
        )

    @copy_docstring(pandas.DataFrame.min)
    def min(
        self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs
    ):
        axis = self._get_axis_number(axis) if axis is not None else 0
        if numeric_only not in (
            None,
            False,
        ):
            raise err._unsupported_error("numeric_only", numeric_only)
        return self._unary_reduction(
            [("min", numeric_only)],
            axis=axis,
            skipna=skipna,
            level=level,
            **kwargs,
        )

    @copy_docstring(pandas.DataFrame.prod)
    def prod(
        self,
        axis=None,
        skipna=None,
        level=None,
        numeric_only=None,
        min_count=0,
        **kwargs,
    ):
        axis = self._get_axis_number(axis) if axis is not None else 0
        if numeric_only not in (
            None,
            True,
        ):
            raise err._unsupported_error("numeric_only", numeric_only)
        if min_count > 0:
            raise err._unsupported_error("min_count", min_count)
        return self._unary_reduction(
            [("prod", numeric_only)],
            axis=axis,
            skipna=skipna,
            level=level,
            min_count=min_count,
            **kwargs,
        )

    @copy_docstring(pandas.DataFrame.sum)
    def sum(
        self,
        axis=None,
        skipna=None,
        level=None,
        numeric_only=None,
        min_count=0,
        **kwargs,
    ):
        axis = self._get_axis_number(axis) if axis is not None else 0
        if numeric_only not in (
            None,
            True,
        ):
            raise err._unsupported_error("numeric_only", numeric_only)
        if min_count > 0:
            raise err._unsupported_error("min_count", min_count)
        return self._unary_reduction(
            [("sum", numeric_only)],
            axis=axis,
            skipna=skipna,
            level=level,
            min_count=min_count,
            **kwargs,
        )

    @copy_docstring(pandas.DataFrame.std)
    def std(
        self,
        axis=None,
        skipna=None,
        level=None,
        ddof=1,
        numeric_only=None,
        **kwargs,
    ):
        axis = self._get_axis_number(axis) if axis is not None else 0
        if numeric_only not in (
            None,
            True,
        ):
            raise err._unsupported_error("numeric_only", numeric_only)
        if ddof != 1:
            raise err._unsupported_error("ddof", ddof)
        return self._unary_reduction(
            [("std", numeric_only)],
            axis=axis,
            skipna=skipna,
            level=level,
            ddof=ddof,
            **kwargs,
        )

    @copy_docstring(pandas.DataFrame.var)
    def var(
        self,
        axis=None,
        skipna=None,
        level=None,
        ddof=1,
        numeric_only=None,
        **kwargs,
    ):
        axis = self._get_axis_number(axis) if axis is not None else 0
        if numeric_only not in (
            None,
            True,
        ):
            raise err._unsupported_error("numeric_only", numeric_only)
        if ddof != 1:
            raise err._unsupported_error("ddof", ddof)
        return self._unary_reduction(
            [("var", numeric_only)],
            axis=axis,
            skipna=skipna,
            level=level,
            ddof=ddof,
            **kwargs,
        )

    product = prod

    def __abs__(self):
        return self.abs()

    def __invert__(self):
        for dtype in self._get_dtypes():
            if not is_numeric_dtype(dtype):
                raise TypeError(f"bad operand type for unary ~: '{dtype}'")
        new_frame = self._frame.unary_op("bit_invert")
        return self._create_or_update_frame(new_frame, False)

    def __neg__(self):
        return self * (-1)

    #############################################
    # Reindexing / selection / label manipulation
    #############################################

    @copy_docstring(pandas.DataFrame.drop)
    def drop(
        self,
        labels=None,
        axis=0,
        index=None,
        columns=None,
        level=None,
        inplace=False,
        errors="raise",
    ):
        # If 'labels' is set, we use 'axis' to determine the lookup axis
        if labels is not None:
            if index is not None or columns is not None:
                raise ValueError(
                    "Cannot specify both 'labels' and 'index'/'columns'"
                )
            axis = self._get_axis_number(axis)

            if axis == 0:
                row_labels = util.to_list_if_scalar(labels)
                row_level = level
                col_labels = []
                col_level = None
            else:
                row_labels = []
                row_level = None
                col_labels = util.to_list_if_scalar(labels)
                col_level = level

        # Otherwise, we use 'columns' and 'index' as lookup labels
        else:
            if not self._is_series and columns is not None:
                col_labels = util.to_list_if_scalar(columns)
                col_level = level
            if index is not None:
                row_labels = util.to_list_if_scalar(index)
                row_level = level

        def _validate_labels(index, labels, level, membership=True):
            for label in labels:
                if not util.is_tuple(label):
                    continue
                if len(label) > index.nlevels:
                    raise KeyError(
                        f"Key length ({len(label)}) exceeds "
                        f"index depth ({index.nlevels})"
                    )

            if not membership:
                return

            if level is not None:
                level = index._get_level_number(level)
                index = index.get_level_values(level)

            for label in labels:
                if label not in index:
                    raise KeyError(label)

        new_self = self.copy(deep=False)

        # Drop columns first as that's easier
        if len(col_labels) > 0:
            assert not new_self._is_series
            _validate_labels(new_self.columns, col_labels, col_level)
            columns = new_self.columns.drop(col_labels, level)
            idxr = new_self.columns.get_indexer_for(columns)
            new_self = new_self._slice_columns(idxr)

        # Then drop rows using selection
        if len(row_labels) > 0:
            _validate_labels(new_self._raw_index, row_labels, row_level, False)

            if len(row_labels) > 1:
                raise err._unsupported_error("Label must be a scalar for now")
            row_label = row_labels[0]

            if level is not None and not is_scalar(row_label):
                raise ValueError("label must be a scalar when 'level' is set")

            if util.is_tuple(row_label) and len(row_label) == 0:
                raise ValueError("label must not be empty")

            mask = new_self._raw_index._get_drop_mask_for(row_label, level)
            new_frame = new_self._frame.select(mask)
            new_self._frame = new_frame

        if inplace:
            if self._is_series:
                self._update_frame(new_self._frame)
            else:
                self._update_frame(new_self._frame, columns=new_self.columns)

        else:
            return new_self

    @copy_docstring(pandas.DataFrame.set_axis)
    def set_axis(self, labels, axis=0, inplace=False):
        axis = self._get_axis_number(axis, 0)

        if axis == 0:
            labels = self._ensure_valid_frame(labels)
            if not labels._is_series:
                raise ValueError("Index data must be 1-dimensional")

            labels = labels._frame._to_index(labels.name)
            return self._create_or_update_frame(
                self._frame.update_legate_index(labels), inplace
            )
        else:
            assert not self._is_series

            if inplace:
                self._replace_columns(labels)
            else:
                new_self = self.copy()
                new_self._replace_columns(labels)
                return new_self

    #######################
    # Missing data handling
    #######################

    @copy_docstring(pandas.DataFrame.dropna)
    def dropna(
        self, axis=0, how="any", thresh=None, subset=None, inplace=False
    ):
        axis = self._get_axis_number(axis, 0)
        inplace = validate_bool_kwarg(inplace, "inplace")

        if axis not in (0,):
            raise err._unsupported_error("axis", axis)

        if how is None and thresh is None:
            raise TypeError("must specify how or thresh")

        if how is not None and how not in ("any", "all"):
            raise ValueError("invalid how option: %s" % how)

        if subset is not None:
            idxr = self.columns.get_indexer_for(subset)
            mask = idxr == -1
            if mask.any():
                raise KeyError(list(np.compress(mask, subset)))
        else:
            idxr = list(range(len(self.columns)))

        if thresh is None:
            thresh = len(idxr) if how == "any" else 1

        new_frame = self._frame.dropna(axis, idxr, thresh)
        return self._create_or_update_frame(new_frame, inplace)

    @copy_docstring(pandas.DataFrame.fillna)
    def fillna(
        self,
        value=None,
        method=None,
        axis=None,
        inplace=False,
        limit=None,
        downcast=None,
    ):
        axis = self._get_axis_number(axis, 0)
        inplace = validate_bool_kwarg(inplace, "inplace")

        if axis not in (0,):
            raise err._unsupported_error("axis", axis)

        if value is None and method is None:
            raise ValueError("must specify a fill method or value")

        if value is not None and method is not None:
            raise ValueError("cannot specify both a fill method and value")

        # Checks on method

        if method is not None:
            raise err._unsupported_error("method", method)

        if method is not None and method not in [
            "backfill",
            "bfill",
            "pad",
            "ffill",
        ]:
            expecting = "pad (ffill) or backfill (bfill)"
            msg = "Invalid fill method. Expecting {expecting}. Got {method}"
            msg = msg.format(expecting=expecting, method=method)
            raise ValueError(msg)

        # Checks on limit

        if limit is not None:
            raise err._unsupported_error("limit", limit)

        if limit is not None:
            if not isinstance(limit, int):
                raise ValueError("Limit must be an integer")
            elif limit <= 0:
                raise ValueError("Limit must be greater than 0")

        # Checks on value

        if isinstance(value, (list, tuple)):
            raise TypeError(
                "'value' parameter must be a scalar or dict, but "
                f"you passed a {type(value).__name__}"
            )

        if is_scalar(value):
            values = {}
            for idx in range(len(self._get_columns())):
                values[idx] = util.sanitize_scalar(value)

        elif is_dict_like(value):
            if self._is_series:
                raise err._unsupported_error(
                    "'value' cannot be a dict for series"
                )

            values = {}
            for col, val in value.items():
                if not is_scalar(val):
                    raise err._unsupported_error(
                        "'value' must be a dict of scalars for now"
                    )
                idxr = self.columns.get_indexer_for([col])
                if idxr[0] != -1:
                    values[idxr[0]] = util.sanitize_scalar(val)

        new_frame = self._frame.fillna(values)
        return self._create_or_update_frame(new_frame, inplace)

    @copy_docstring(pandas.DataFrame.isna)
    def isna(self):
        return self._create_or_update_frame(self._frame.isna(), False)

    isnull = isna

    @copy_docstring(pandas.DataFrame.notna)
    def notna(self):
        return self._create_or_update_frame(self._frame.notna(), False)

    notnull = notna

    #################################
    # Reshaping, sorting, transposing
    #################################

    @copy_docstring(pandas.DataFrame.droplevel)
    def droplevel(self, level, axis=0):
        axis = self._get_axis_number(axis)
        new_self = self.copy()
        if axis == 1:
            new_self.columns = new_self.columns.droplevel(level)
        else:
            new_self._frame = new_self._frame.droplevel(level)
        return new_self

    @staticmethod
    def _get_ascending(ascending, num):
        if isinstance(ascending, list):
            return [bool(asc) for asc in ascending]
        else:
            return [bool(ascending)] * num

    @copy_docstring(pandas.DataFrame.sort_values)
    def sort_values(
        self,
        by,
        axis=0,
        ascending=True,
        inplace: bool = False,
        kind="quicksort",
        na_position="last",
        ignore_index: bool = False,
    ):
        axis = self._get_axis_number(axis)
        if axis not in (0,):
            raise err._unsupported_error("axis", axis)

        if na_position not in (
            "first",
            "last",
        ):
            raise err._invalid_value_error("na_position", na_position)

        by = util.to_list_if_scalar(by)
        ascending = self._get_ascending(ascending, len(by))
        if len(by) != len(ascending):
            raise ValueError(
                f"Length of ascending ({len(ascending)}) != "
                f"length of by ({len(by)})"
            )

        idxr = self.columns.get_indexer_for(by)
        if len(idxr) != len(by):
            for key in by:
                if len(by.count(key)) > 1:
                    raise ValueError("The column label '{key}' is not unique.")

        new_frame = self._frame.sort_values(
            idxr,
            axis,
            ascending,
            kind,
            na_position,
            ignore_index,
        )
        return self._create_or_update_frame(new_frame, inplace)

    @copy_docstring(pandas.DataFrame.sort_index)
    def sort_index(
        self,
        axis=0,
        level=None,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
        sort_remaining=True,
        ignore_index: bool = False,
    ):
        axis = self._get_axis_number(axis)
        if axis not in (0,):
            raise err._unsupported_error("axis", axis)

        nlevels = self._raw_index.nlevels
        if nlevels == 1:
            # Pandas ignores level and sort_remaining for single-level indices,
            levels = [0] if level is None else util.to_list_if_scalar(level)
            # and it casts ascending to a boolean value...
            ascending = [bool(ascending)]
        else:
            if level is None:
                levels = list(range(nlevels))
                # When level is None, Pandas crops the ascending list
                # to match its length to the number of levels...
                ascending = self._get_ascending(ascending, nlevels)[:nlevels]
            else:
                levels = util.to_list_if_scalar(level)
                levels = [
                    self._raw_index._get_level_number(lvl) for lvl in levels
                ]
                default_asc = bool(ascending)
                ascending = self._get_ascending(ascending, len(levels))
                if len(ascending) != len(levels):
                    raise ValueError(
                        "level must have same length as ascending"
                    )
                # XXX: Pandas ignores sort_remaining for multi-level indices
                #      (GH #24247), and always sorts the levels monotonically
                #      before the actual sorting...
                #      Here we do the right thing and hopefully Pandas fixes
                #      its bug in the future.
                if sort_remaining:
                    already_added = set(levels)
                    for lvl in range(nlevels):
                        if lvl not in already_added:
                            levels.append(lvl)
                            ascending.append(default_asc)

        new_frame = self._frame.sort_index(
            axis=axis,
            levels=levels,
            ascending=ascending,
            kind=kind,
            na_position=na_position,
            ignore_index=ignore_index,
        )
        return self._create_or_update_frame(new_frame, inplace)

    ###########################################
    # Combining / comparing / joining / merging
    ###########################################

    @copy_docstring(pandas.DataFrame.append)
    def append(
        self, other, ignore_index=False, verify_integrity=False, sort=False
    ):
        from .utils import concat

        return concat(
            [self, other],
            axis=0,
            ignore_index=ignore_index,
            verify_integrity=verify_integrity,
            sort=sort,
        )

    #################################
    # Serialization / IO / conversion
    #################################

    @copy_docstring(pandas.DataFrame.to_csv)
    def to_csv(
        self,
        path_or_buf=None,
        sep=",",
        na_rep="",
        columns=None,
        header=True,
        index=True,
        line_terminator=None,
        chunksize=None,
        partition=False,
    ):
        if not isinstance(path_or_buf, str):
            raise err._unsupported_error("path must be a string for now")

        if len(sep) != 1:
            raise err._unsupported_error("separator must be a character")

        line_terminator = (
            os.linesep if line_terminator is None else line_terminator
        )

        # The default chunk size is 8
        chunksize = 8 if chunksize is None else chunksize

        new_self = self
        if columns is not None:
            new_self = self[util.to_list_if_scalar(columns)]

        new_self._frame.to_csv(
            path=path_or_buf,
            sep=sep,
            na_rep=na_rep,
            header=header,
            index=index,
            line_terminator=line_terminator,
            chunksize=chunksize,
            partition=partition,
            column_names=new_self.columns.to_list(),
        )

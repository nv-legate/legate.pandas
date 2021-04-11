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
import warnings

import pandas
from pandas.api.types import is_dict_like, is_integer, is_list_like, is_scalar
from pandas.core.common import is_bool_indexer
from pandas.core.indexes.api import ensure_index as pandas_ensure_index

from legate.pandas.common import errors as err, util as util
from legate.pandas.core.index import BaseIndex

from .doc_utils import copy_docstring
from .frame import Frame
from .import_utils import from_named_legate_data, from_pandas
from .modin_utils import _repr_dataframe
from .series import Series


def _is_pandas_container(obj):
    return isinstance(obj, (pandas.DataFrame, pandas.Series))


class DataFrame(Frame):
    _INTERNAL_MEMBERS = {"_frame", "_columns"}

    def __init__(
        self,
        data=None,
        index=None,
        columns=None,
        dtype=None,
        copy=False,
        frame=None,
    ):
        # TODO: We would want to hide the frame argument from the users,
        #       as it is intended only for internal uses
        if frame is not None:
            assert index is None
            assert dtype is None
            assert columns is not None
            assert len(columns) == len(frame._columns)
            self._frame = frame
            self._set_columns(columns)

        elif isinstance(data, type(self)):
            self._construct_from_dataframe(data, index, columns, dtype, copy)

        elif isinstance(data, Frame):
            self._construct_from_series(data, index, columns, dtype, copy)

        elif (
            not _is_pandas_container(data)
            and is_dict_like(data)
            and len(data) > 0
        ):
            if all(isinstance(val, Frame) for val in data.values()):
                self._construct_from_frames(data, index, columns, dtype, copy)
            elif all(
                hasattr(val, "__legate_data_interface__")
                for val in data.values()
            ):
                self._construct_from_legate_containers(
                    data, index, columns, dtype, copy
                )
            else:
                self._construct_fallback(data, index, columns, dtype, copy)

        else:
            self._construct_fallback(data, index, columns, dtype, copy)

        assert self._columns is not None

    def _construct_from_dataframe(self, data, index, columns, dtype, copy):
        if index is not None:
            raise err._unsupported_error(
                "index can be used only when importing "
                "Pandas dataframes or series"
            )
        # When columns are given, we should project the frame onto them
        if columns is not None:
            # TODO: For now we use Pandas for validation, as we keep
            #       columns in a Pandas index. Once we replace it with
            #       our own index type, this check needs to be replaced
            #       as well.
            columns = pandas_ensure_index(columns)
            missing_columns = []
            for column in columns:
                if column not in data.columns:
                    missing_columns.append(column)

            if len(missing_columns) > 0:
                raise ValueError(
                    f"None of {missing_columns} exist "
                    f"in the input dataframe"
                )
            data = data[columns]
        else:
            columns = data.columns

        # When a dtype is given, the source dataframe must be casted
        # before we copy its frame
        if dtype is not None:
            data = data.astype(dtype)

        assert len(columns) == len(data._frame._columns)
        self._frame = data._frame.copy() if copy else data._frame
        self._set_columns(columns)

    def _construct_from_series(self, data, index, columns, dtype, copy):
        assert data._is_series
        if index is not None:
            raise err._unsupported_error(
                "index can be used only when importing "
                "Pandas dataframes or series"
            )

        # TODO: For now we only allow indices of length 1 for the columns
        if columns is not None:
            columns = pandas_ensure_index(columns)
            if len(columns) != 1:
                raise ValueError("the length of columns must be 1")
            if data.name is not None and data.name not in columns:
                raise ValueError(f"column {columns[0]} does not exist")
        else:
            columns = data._get_columns()

        # When a dtype is given, the source dataframe must be casted
        # before we copy its frame
        if dtype is not None:
            data = data.astype(dtype)

        self._frame = data._frame.copy() if copy else data._frame
        self._set_columns(columns)

    def _construct_from_legate_containers(
        self, data, index, columns, dtype, copy
    ):
        if index is not None:
            raise err._unsupported_error(
                "index cannot be used when importing Legate Data"
            )

        if columns is not None:
            raise err._unsupported_error(
                "columns cannot be used when importing Legate Data"
            )

        if dtype is not None:
            raise err._unsupported_error(
                "dtype cannot be used when importing Legate Data"
            )

        self._frame, columns = from_named_legate_data(data)

        if copy:
            self._frame = self._frame.copy()

        self._set_columns(columns)

    def _construct_from_frames(self, data, index, columns, dtype, copy):
        if index is not None:
            raise err._unsupported_error(
                "index cannot be used when importing Legate frames"
            )

        if columns is not None:
            raise err._unsupported_error(
                "columns cannot be used when importing Legate frames"
            )

        if any(not val._is_series for val in data.values()):
            raise ValueError("dictionary values must be all series")

        # TODO: Here we need to join the indices and reindex the series
        #       using that joined index. Since we haven't implemented
        #       reindex, we will align the first series with the others
        #       just to make sure that they are aligned (_align_frame
        #       is currently doing nothing more than checking that
        #       the indices are the same).

        all_series = list(data.values())
        first = all_series[0]
        others = all_series[1:]
        others = [first._align_frame(other, axis=0)[1] for other in others]

        others_frames = [other._frame for other in others]
        self._frame = first._frame.concat(1, others_frames)

        if copy:
            self._frame = self._frame.copy()
        self._set_columns(list(data.keys()))

    def _construct_fallback(self, data, index, columns, dtype, copy):
        # If the index is a Legate index, we reuse its storage when
        # importing the Pandas dataframe
        if isinstance(index, BaseIndex):
            tmp_df = pandas.DataFrame(
                data=data,
                columns=columns,
                dtype=dtype,
                copy=copy,
            )
            self._frame = from_pandas(tmp_df, index=index)

        # Otherwise, we construct the complete Pandas dataframe
        # and import it
        else:
            tmp_df = pandas.DataFrame(
                data=data,
                index=index,
                columns=columns,
                dtype=dtype,
                copy=copy,
            )
            self._frame = from_pandas(tmp_df)
        self._set_columns(tmp_df.columns)

    ###########
    # Utilities
    ###########

    @property
    def _is_series(self):
        return False

    def _create_or_update_frame(self, new_frame, inplace, columns=None):
        if inplace:
            self._frame = new_frame
            if columns is not None:
                self.columns = columns
            return None

        else:
            columns = self.columns if columns is None else columns
            new_self = self.__ctor__(frame=new_frame, columns=columns)
            return new_self

    def _reduce_dimension(self, frame):
        return Series(frame=frame)

    def _shuffle(self, keys):
        keys = util.to_list_if_scalar(keys)
        idxr = self.columns.get_indexer_for(keys)
        return self._create_or_update_frame(self._frame._shuffle(idxr), False)

    ################################
    # Attributes and underlying data
    ################################

    def __repr__(self):
        return _repr_dataframe(self)

    def _get_columns(self):
        return self._columns

    def _set_columns(self, columns):
        if not isinstance(columns, pandas.Index):
            columns = pandas.Index(columns)
        if hasattr(self, "_columns"):
            old_len = len(self._columns)
            new_len = len(columns)
            if old_len != new_len:
                raise ValueError(
                    f"Length mismatch: Expected axis has {old_len} elements, "
                    f"new values have {new_len} elements"
                )

        self._columns = columns

    def _replace_columns(self, new_columns):
        del self._columns
        self._set_columns(new_columns)

    columns = property(
        _get_columns, _set_columns, None, pandas.DataFrame.columns.__doc__
    )

    @copy_docstring(pandas.DataFrame.dtypes)
    @property
    def dtypes(self):
        return pandas.Series(self._frame.dtypes, index=self.columns)

    @copy_docstring(pandas.DataFrame.axes)
    @property
    def axes(self):
        return [self.index, self.columns]

    @copy_docstring(pandas.DataFrame.ndim)
    @property
    def ndim(self):
        return 2

    @copy_docstring(pandas.DataFrame.shape)
    @property
    def shape(self):
        return (len(self), len(self.columns))

    ############
    # Conversion
    ############

    def copy(self, deep=True):
        if deep:
            return self.__ctor__(
                frame=self._frame.copy(), columns=self.columns
            )
        return self.__ctor__(frame=self._frame, columns=self.columns)

    #####################
    # Indexing, iteration
    #####################

    def _slice_columns(self, col_indexer):
        columns = self.columns[col_indexer]
        frame = self._frame.slice_columns(col_indexer)
        return self.__ctor__(frame=frame, columns=columns)

    def _get_columns_by_labels(self, key):
        key_scalar = is_scalar(key) or isinstance(key, tuple)
        keys = util.to_list_if_scalar(key)
        columns = self.columns

        # Validate keys
        for key in keys:
            if key not in columns:
                raise KeyError(key)

        indexer = columns.get_indexer_for(keys)
        new_self = self._slice_columns(indexer)
        if key_scalar:
            assert len(new_self.columns) == 1
            return new_self.squeeze(axis=1)
        else:
            return new_self

    def _set_columns_by_labels(self, key, item):
        keys = util.to_list_if_scalar(key)
        columns = self.columns

        # Validate keys
        found = []
        fresh = []
        for key in keys:
            if key in columns:
                found.append(key)
            else:
                fresh.append(key)

        # TODO: for now we disallow insertions mixed with inplace updates
        if len(found) > 0 and len(fresh) > 0:
            raise err._unsupported_error(
                "In-place updates cannot be mixed with insertions. "
                "Please split them into multiple statements."
            )

        if not is_scalar(item):
            item = self._ensure_valid_frame(item)
            _, item = self._align_frame(item, join="left", axis=0)

            if item._is_series:
                if len(keys) > 1:
                    raise err._unsupported_error(
                        "Broadcasting a series to multiple columns is "
                        "not yet supported"
                    )
            else:
                if len(keys) != len(item.columns):
                    raise ValueError("Columns must be same length as key")

        if len(found) > 0:
            indexer = columns.get_indexer_for(found)
            if is_scalar(item):
                item = self._frame.create_column_from_scalar(item)
                item = item.broadcast(len(indexer))
            else:
                item = item._frame

            self._frame.update_columns(indexer, item)

        else:
            if is_scalar(item):
                for _ in range(len(fresh)):
                    idx = self._frame.num_columns()
                    self._frame = self._frame.insert(idx, item)
            else:
                item = DataFrame(frame=item._frame, columns=fresh)
                self._frame = self._frame.concat(1, item._frame)
            self._replace_columns(columns.append(pandas.Index(fresh)))

    @copy_docstring(pandas.DataFrame.at)
    @property
    def at(self):
        from .indexing import _LocDataframeLocator

        return _LocDataframeLocator(self, is_at=True)

    @copy_docstring(pandas.DataFrame.iat)
    @property
    def iat(self):
        from .indexing import _IlocDataframeLocator

        return _IlocDataframeLocator(self, is_at=True)

    @copy_docstring(pandas.DataFrame.loc)
    @property
    def loc(self):
        from .indexing import _LocDataframeLocator

        return _LocDataframeLocator(self)

    @copy_docstring(pandas.DataFrame.iloc)
    @property
    def iloc(self):
        from .indexing import _IlocDataframeLocator

        return _IlocDataframeLocator(self)

    @copy_docstring(pandas.DataFrame.insert)
    def insert(self, loc, column, value, allow_duplicates=False):
        if not is_integer(loc):
            raise TypeError("'loc' must be an integer")

        elif loc < 0:
            raise ValueError("unbounded slice")

        elif loc > len(self.columns):
            raise IndexError(
                f"index {loc} is out of bounds for axis 0 with "
                f"size {len(self.columns)}"
            )

        elif not allow_duplicates and column in self.columns:
            raise ValueError(f"cannot insert {column}, already exists")

        value = self._ensure_valid_frame(value)

        if not is_scalar(value):
            if not value._is_series and len(value.columns) != 1:
                raise ValueError(
                    "Wrong number of items passed 2, placement implies 1"
                )

            _, value = self._align_frame(value, join="left", axis=0)
            value = value._frame

        if self.empty and self._raw_index is None:
            if is_scalar(value):
                frame = DataFrame(columns=[column])._frame
                self._update_frame(frame)
            else:
                self._update_frame(value)
            self._replace_columns([column])
        else:
            self._update_frame(self._frame.insert(loc, value))
            self._replace_columns(self.columns.insert(loc, column))

    @copy_docstring(pandas.DataFrame.keys)
    def keys(self):
        return self.columns

    @copy_docstring(pandas.DataFrame.where)
    def where(
        self,
        cond,
        other=None,
        inplace=False,
        axis=None,
        level=None,
        errors="raise",
        try_cast=False,
    ):
        """
        Replace values where the condition is False.

        Parameters
        ----------
        cond : bool Series/DataFrame, array-like
            Where cond is True, keep the original value.
            Where False, replace with corresponding value from other.
            Callables are not supported.
        other: scalar, list of scalars, Series/DataFrame
            Entries where cond is False are replaced with
            corresponding value from other. Callables are not
            supported. Default is None.

            DataFrame expects only Scalar or array like with scalars or
            dataframe with same dimension as self.

            Series expects only scalar or series like with same length
        inplace : bool, default False
            Whether to perform the operation in place on the data.

        Returns
        -------
        Same type as caller
        """

        return self._copy_if_else(
            cond,
            other=other,
            inplace=inplace,
            axis=axis,
            level=level,
            errors=errors,
            try_cast=try_cast,
            negate=False,
        )

    @copy_docstring(pandas.DataFrame.mask)
    def mask(
        self,
        cond,
        other=None,
        inplace=False,
        axis=None,
        level=None,
        errors="raise",
        try_cast=False,
    ):
        """
        Replace values where the condition is True.

        Parameters
        ----------
        cond : bool Series/DataFrame, array-like
            Where cond is False, keep the original value.
            Where True, replace with corresponding value from other.
            Callables are not supported.
        other: scalar, list of scalars, Series/DataFrame
            Entries where cond is True are replaced with
            corresponding value from other. Callables are not
            supported. Default is None.

            DataFrame expects only Scalar or array like with scalars or
            dataframe with same dimension as self.

            Series expects only scalar or series like with same length
        inplace : bool, default False
            Whether to perform the operation in place on the data.

        Returns
        -------
        Same type as caller
        """

        return self._copy_if_else(
            cond,
            other=other,
            inplace=inplace,
            axis=axis,
            level=level,
            errors=errors,
            try_cast=try_cast,
            negate=True,
        )

    @copy_docstring(pandas.DataFrame.query)
    def query(self, expr, inplace=False, **kwargs):
        if inplace not in (
            True,
            False,
        ):
            raise err._invalid_value_error("inplace", inplace)
        if not isinstance(expr, str):
            msg = f"expr must be a string to be evaluated, {type(expr)} given"
            raise ValueError(msg)
        if not expr:
            raise ValueError("expr cannot be an empty string")
        new_frame = self._frame.query(self.columns, expr, **kwargs)
        return self._create_or_update_frame(new_frame, inplace)

    def __getattr__(self, key):
        if key in self._INTERNAL_MEMBERS:
            return object.__getattribute__(self, key)
        else:
            if key in self.columns:
                return self[key]

        if hasattr(pandas.DataFrame, key):
            raise err._unsupported_error(
                f"DataFrame.{key} is not yet implemented in Legate Pandas."
            )
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute {key}"
        )

    def __setattr__(self, key, value):
        try:
            object.__getattribute__(self, key)
            object.__setattr__(self, key, value)
            return
        except AttributeError:
            pass

        if key in self._INTERNAL_MEMBERS:
            object.__setattr__(self, key, value)
        else:
            self[key] = value

    def __getitem__(self, key):
        if is_scalar(key) or isinstance(key, tuple):
            return self._get_columns_by_labels(key)
        elif isinstance(key, slice):
            return self.iloc[key]
        elif isinstance(key, (DataFrame, pandas.DataFrame)):
            return self.where(key)
        elif isinstance(key, Series):
            return self.loc[key]
        elif is_list_like(key):
            if is_bool_indexer(key):
                return self.loc[key]
            else:
                return self._get_columns_by_labels(key)
        else:
            raise ValueError(f"Unsupported key type '{type(key).__name}'")

    def __setitem__(self, key, value):
        if is_scalar(key) or isinstance(key, tuple):
            self._set_columns_by_labels(key, value)
        elif isinstance(key, slice):
            self.iloc[key] = value
        elif isinstance(key, Series):
            self.loc[key] = value
        elif is_list_like(key):
            if is_bool_indexer(key):
                self.loc[key] = value
            else:
                self._set_columns_by_labels(key, value)
        # TODO: Pandas supports in-place updates with boolean masks
        #       stored in dataframes (which, surprisingly, is not
        #       supported via loc for some reason). Since we delgate
        #       in-place updates to loc/iloc in this method, we will
        #       disallow boolean dataframes as keys for now.
        # elif isinstance(key, (DataFrame, pandas.DataFrame)):
        else:
            raise ValueError(f"Unsupported key type '{type(key).__name}'")

    def __iter__(self):
        return iter(self.columns)

    def __contains__(self, key):
        return key in self.columns

    def __delitem__(self, key):
        if key not in self:
            raise KeyError(key)

        new_columns = []
        idxr = []
        for idx, col in enumerate(self.columns):
            if col != key:
                new_columns.append(col)
                idxr.append(idx)

        self._replace_columns(pandas.Index(new_columns))
        self._update_frame(self._frame.slice_columns(idxr))

    ###########################
    # Binary operator functions
    ###########################

    def _align_frame(
        self,
        other,
        join="outer",
        axis=None,
        fill_value=None,
        broadcast_axis=None,
    ):
        lhs, rhs = self, other

        axis = lhs._get_axis_number(axis)
        axes = (0, 1) if axis is None else (axis,)

        if 0 in axes:
            rhs = rhs._create_or_update_frame(
                lhs._raw_index._align_partition(rhs._frame), False
            )

            if not lhs._raw_index.equals(rhs._raw_index):
                raise err._unsupported_error(
                    "Unaligend dataframes are not supported yet"
                )

        if rhs._is_series:
            if broadcast_axis == 1:
                rhs = rhs._broadcast(lhs.columns)

        elif 1 in axes:
            joined, idxr1, idxr2 = lhs.columns.join(
                rhs.columns,
                how=join,
                return_indexers=True,
            )

            if idxr1 is not None:
                rhs_dtypes = rhs._frame._get_dtypes()
                frame = lhs._frame.align_columns(idxr1, fill_value, rhs_dtypes)
                lhs = DataFrame(frame=frame, columns=joined)
            if idxr2 is not None:
                lhs_dtypes = lhs._frame._get_dtypes()
                frame = rhs._frame.align_columns(idxr2, fill_value, lhs_dtypes)
                rhs = DataFrame(frame=frame, columns=joined)

        return lhs, rhs

    @copy_docstring(pandas.DataFrame.add)
    def add(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "add", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.DataFrame.floordiv)
    def floordiv(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "floordiv", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.DataFrame.mod)
    def mod(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "mod", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.DataFrame.mul)
    def mul(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "mul", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.DataFrame.pow)
    def pow(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "pow", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.DataFrame.sub)
    def sub(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "sub", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.DataFrame.truediv)
    def truediv(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "truediv", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.DataFrame.rfloordiv)
    def rfloordiv(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "rfloordiv", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.DataFrame.rmod)
    def rmod(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "rmod", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.DataFrame.rpow)
    def rpow(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "rpow", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.DataFrame.rsub)
    def rsub(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "rsub", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.DataFrame.rtruediv)
    def rtruediv(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "rtruediv", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.DataFrame.eq)
    def eq(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op("eq", other, axis=axis, level=level)

    @copy_docstring(pandas.DataFrame.le)
    def le(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op("le", other, axis=axis, level=level)

    @copy_docstring(pandas.DataFrame.lt)
    def lt(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op("lt", other, axis=axis, level=level)

    @copy_docstring(pandas.DataFrame.ge)
    def ge(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op("ge", other, axis=axis, level=level)

    @copy_docstring(pandas.DataFrame.gt)
    def gt(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op("gt", other, axis=axis, level=level)

    @copy_docstring(pandas.DataFrame.ne)
    def ne(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op("ne", other, axis=axis, level=level)

    def __and__(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "__and__", other, axis=axis, level=level, fill_value=fill_value
        )

    def __or__(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "__or__", other, axis=axis, level=level, fill_value=fill_value
        )

    def __xor__(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "__xor__", other, axis=axis, level=level, fill_value=fill_value
        )

    div = divide = truediv
    rmul = multiply = mul
    radd = add
    rdiv = rtruediv
    subtract = sub

    __add__ = add
    __div__ = div
    __floordiv__ = floordiv
    __mul__ = mul
    __pow__ = pow
    __sub__ = sub
    __truediv__ = truediv
    __mod__ = mod

    __eq__ = eq
    __ge__ = ge
    __gt__ = gt
    __le__ = le
    __lt__ = lt
    __ne__ = ne

    __rmod__ = rmod
    __radd__ = radd
    __rdiv__ = rdiv
    __rfloordiv__ = rfloordiv
    __rmul__ = rmul
    __rpow__ = rpow
    __rsub__ = rsub
    __rtruediv__ = rtruediv

    __iadd__ = __add__
    __imul__ = __mul__
    __ipow__ = __pow__
    __isub__ = __sub__
    __ifloordiv__ = __floordiv__
    __itruediv__ = __truediv__
    __imod__ = __mod__

    ########################################
    # Function application, GroupBy & window
    ########################################

    @copy_docstring(pandas.DataFrame.groupby)
    def groupby(
        self,
        by=None,
        axis=0,
        level=None,
        as_index=True,
        sort=False,
        **kwargs,
    ):
        from .groupby import DataFrameGroupBy

        return DataFrameGroupBy(
            df=self,
            by=by,
            axis=axis,
            level=level,
            as_index=as_index,
            sort=sort,
            method=kwargs.get("method", "hash"),
        )

    ##################################
    # Computations / descriptive stats
    ##################################

    #############################################
    # Reindexing / selection / label manipulation
    #############################################

    @copy_docstring(pandas.DataFrame.add_prefix)
    def add_prefix(self, prefix):
        def _add_prefix(label):
            if isinstance(label, tuple):
                return tuple(map(_add_prefix, label))
            else:
                return prefix + str(label)

        return self.__ctor__(
            frame=self._frame, columns=self.columns.map(_add_prefix)
        )

    @copy_docstring(pandas.DataFrame.add_suffix)
    def add_suffix(self, suffix):
        def _add_suffix(label):
            if isinstance(label, tuple):
                return tuple(map(_add_suffix, label))
            else:
                return str(label) + suffix

        return self.__ctor__(
            frame=self._frame, columns=self.columns.map(_add_suffix)
        )

    @copy_docstring(pandas.DataFrame.equals)
    def equals(self, other):
        if not isinstance(other, DataFrame):
            other = DataFrame(other)
        return self.columns.equals(other.columns) and self._frame.equals(
            other._frame
        )

    @copy_docstring(pandas.DataFrame.rename)
    def rename(
        self,
        mapper=None,
        index=None,
        columns=None,
        axis=None,
        copy=True,
        inplace=False,
        level=None,
        errors="ignore",
    ):
        axis = self._get_axis_number(axis)

        # Check the spec with an empty Pandas object
        self._get_empty_pandas_object().rename(
            mapper=mapper,
            index=index,
            columns=columns,
            axis=axis,
            copy=copy,
            inplace=inplace,
            level=level,
            errors=errors,
        )

        # Filter out the cases that are not yet supported
        if index is not None or (mapper is not None and axis == 0):
            raise err._unsupported_error("renaming index is not supported yet")

        if level not in (None,):
            raise err._unsupported_error("level", level)

        # Use a mock Pandas dataframe to rename column names
        tmp_df = pandas.DataFrame(columns=self.columns).rename(
            mapper=mapper, columns=columns, axis=axis
        )
        new_columns = tmp_df.columns

        new_self = self if inplace else self.copy(deep=copy)
        new_self.columns = new_columns

        if not inplace:
            return new_self

    @copy_docstring(pandas.DataFrame.reset_index)
    def reset_index(
        self, level=None, drop=False, inplace=False, col_level=0, col_fill=""
    ):
        if inplace not in (
            True,
            False,
        ):
            raise err._invalid_value_error("inplace", inplace)
        if drop not in (
            True,
            False,
        ):
            raise err._invalid_value_error("drop", drop)

        if level is None:
            levels = list(range(self._raw_index.nlevels))
        else:
            levels = util.to_list_if_scalar(level)
            levels = [self._raw_index._get_level_number(lvl) for lvl in levels]
        # Pandas seems to ignore the order in which the levels are specified
        # but rather sorts them
        levels = sorted(levels)

        frame = self._frame.reset_index(levels, drop)
        columns = self.columns
        # FIXME: For now we will ignore the corner case where a column
        #        named index or level_0 already exists.
        if not drop:
            names = self._raw_index._get_level_names(levels)

            lev_num = columns._get_level_number(col_level)
            if isinstance(columns, pandas.MultiIndex):
                arrays = [[col_fill] * len(names)] * columns.nlevels
                arrays[lev_num] = names
                names = pandas.MultiIndex.from_arrays(arrays)
            else:
                names = pandas.Index(names)

            columns = names.append(columns)

        return self._create_or_update_frame(frame, inplace, columns=columns)

    @copy_docstring(pandas.DataFrame.set_index)
    def set_index(
        self,
        keys,
        drop=True,
        append=False,
        inplace=False,
        verify_integrity=False,
    ):
        if inplace not in (
            True,
            False,
        ):
            raise err._invalid_value_error("inplace", inplace)
        keys = util.to_list_if_scalar(keys)
        keys = [
            Series(key) if not isinstance(key, (str, Series)) else key
            for key in keys
        ]

        frame = self._frame
        columns = self.columns

        missing = []
        to_drop = []
        to_set = []
        names = []
        if append:
            to_set = util.to_list_if_scalar(self._raw_index.column)
            names = util.to_list_if_scalar(self._raw_index.name)

        for key in keys:
            if not isinstance(key, Series):
                if key in columns:
                    idxr = columns.get_indexer_for([key])
                    to_drop.extend(idxr)
                    to_set.extend(self._frame.select_columns(idxr))
                    names.append(key)
                else:
                    missing.append(key)
            else:
                new_len = len(key)
                old_len = len(self)
                if new_len != old_len:
                    raise ValueError(
                        f"Length mismatch: Expected {old_len} rows, "
                        f"received array of length {new_len}"
                    )
                to_set.append(key._frame._columns[0])
                names.append(key.name)

        if missing:
            raise KeyError(f"None of {missing} are in the columns")

        if drop:
            columns = columns.delete(to_drop)
            frame = frame.drop_columns(to_drop)

        frame = frame.set_index(to_set, names)
        return DataFrame(frame=frame, columns=columns)

    #######################
    # Missing data handling
    #######################

    #################################
    # Reshaping, sorting, transposing
    #################################

    @copy_docstring(pandas.DataFrame.squeeze)
    def squeeze(self, axis=None):
        axis = self._get_axis_number(axis, None)
        if axis not in (
            1,
            None,
        ):
            raise err._unsupported_error("axis", axis)

        result = self

        if (
            axis
            in (
                1,
                None,
            )
            and len(result.columns) == 1
        ):
            result = Series(frame=result._frame, name=result.columns[0])

        if (
            axis
            in (
                0,
                None,
            )
            and len(result) == 1
        ):
            if result._is_series:
                result = result.to_pandas().squeeze()
            else:
                # TODO: We want to handle this case once we support series
                #       of mixed type values (which would be either expressed
                #       by its transpose or backed by a Pandas series).
                warnings.warn(
                    "Squeezing a dataframe on both axes is currently "
                    "unsupported unless the size is 1. Squeeze for axis=0 "
                    "will be ignored."
                )

        return result

    ###########################################
    # Combining / comparing / joining / merging
    ###########################################

    @copy_docstring(pandas.DataFrame.join)
    def join(
        self,
        other,
        on=None,
        how="left",
        lsuffix="",
        rsuffix="",
        sort=False,
        **kwargs,
    ):
        if not isinstance(other, (Frame, pandas.DataFrame, pandas.Series)):
            raise ValueError("other must be a DataFrame or a Series")

        other = self._ensure_valid_frame(other)

        kwargs["how"] = how
        kwargs["suffixes"] = (lsuffix, rsuffix)
        kwargs["sort"] = sort
        kwargs["right_index"] = True

        if on is not None:
            kwargs["left_on"] = on
        else:
            if other._is_series and other.name is None:
                raise ValueError("Other Series must have a name")

            kwargs["left_index"] = True

        return self.merge(other, **kwargs)

    @copy_docstring(pandas.DataFrame.merge)
    def merge(
        self,
        right,
        how="inner",
        on=None,
        left_on=None,
        right_on=None,
        left_index=False,
        right_index=False,
        sort=False,
        suffixes=("_x", "_y"),
        copy=True,
        **kwargs,
    ):
        from .merge import Merge

        merge = Merge(
            self,
            right,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            sort=sort,
            suffixes=suffixes,
            copy=copy,
            **kwargs,
        )

        return merge.perform_merge()

    #################################
    # Serialization / IO / conversion
    #################################

    def to_pandas(self, schema_only=False):
        """
        Convert distributed DataFrame into a Pandas DataFrame

        Parameters
        ----------
        schema_only : Doesn't convert the data when True

        Returns
        -------
        out : pandas.DataFrame
        """

        df = self._frame.to_pandas(schema_only=schema_only)
        df.columns = self.columns
        return df

    @copy_docstring(pandas.DataFrame.to_parquet)
    def to_parquet(
        self,
        path,
        engine="auto",
        compression="snappy",
        index=None,
        partition_cols=None,
        **kwargs,
    ):
        if not isinstance(path, str):
            raise err._unsupported_error("path must be a string for now")

        if os.path.exists(path):
            raise ValueError(f"{path} already exists")

        if partition_cols is not None:
            raise err._unsupported_error("partition_cols", partition_cols)

        if compression not in (
            "snappy",
            None,
        ):
            raise err._unsupported_error("compression", compression)

        if any(not isinstance(col, str) for col in self.columns):
            raise ValueError("parquet must have string column names")

        self._frame.to_parquet(
            path,
            self.columns.to_list(),
            engine=engine,
            compression=compression,
            index=index,
            partition_cols=partition_cols,
            **kwargs,
        )

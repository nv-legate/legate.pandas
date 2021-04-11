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

import pandas
from pandas._libs import lib as lib
from pandas.api.types import is_hashable

from legate.pandas.common import errors as err, util as util

from .accessors import CategoricalAccessor, DatetimeProperties, StringMethods
from .doc_utils import copy_docstring
from .frame import Frame
from .import_utils import from_legate_data, from_pandas
from .modin_utils import _repr_series


class Series(Frame):
    def __init__(
        self,
        data=None,
        index=None,
        dtype=None,
        name=None,
        copy=False,
        frame=None,
    ):
        """
        One-dimensional distributed array with axis labels.

        Labels need not be unique but must be a hashable type. The object
        supports both integer- and label-based indexing and provides a
        host of methods for performing operations involving the index.
        Statistical methods from ndarray have been overridden to
        automatically exclude missing data (currently represented
        as null/NaN).

        Operations between Series (`+`, `-`, `/`, `*`, `**`) align
        values based on their associated index values-– they need
        not be the same length. The result index will be the
        sorted union of the two indexes.

        ``Series`` objects are used as columns of ``DataFrame``.

        Parameters
        ----------
        data : array-like, Iterable, dict, or scalar value
            Contains data stored in Series.

        index : array-like or Index (1d)
            Values must be hashable and have the same length
            as data. Non-unique index values are allowed. Will
            default to RangeIndex (0, 1, 2, …, n) if not provided.
            If both a dict and index sequence are used, the index will
            override the keys found in the dict.

        dtype : str, numpy.dtype, or ExtensionDtype, optional
            Data type for the output Series. If not specified,
            this will be inferred from data.

        name : str, optional
            The name to give to the Series.

        nan_as_null : bool, Default True
            If ``None``/``True``, converts ``np.nan`` values to
            ``null`` values.
            If ``False``, leaves ``np.nan`` values as is.

        frame : Table
            Storage manager object used for internal purposes only
        """
        self._name = None
        if frame is not None:
            assert data is None
            assert index is None
            assert dtype is None
            self._frame = frame
            self._set_name(name)

        elif isinstance(data, type(self)):
            if index is not None:
                raise err._unsupported_error(
                    "index can be used only when importing "
                    "Pandas dataframes or series"
                )

            if dtype is not None:
                data = data.astype(dtype)

            self._frame = data._frame.copy() if copy else data._frame
            self._set_name(self.name if name is None else name)

        elif hasattr(data, "__legate_data_interface__"):
            if index is not None:
                raise err._unsupported_error(
                    "index cannot be used when importing a Legate Data"
                )

            if dtype is not None:
                raise err._unsupported_error(
                    "dtype cannot be used when importing a Legate Data"
                )

            self._frame = from_legate_data(data.__legate_data_interface__)
            self._set_name(name)

        else:
            from legate.pandas.core.index import BaseIndex

            # If the index is a Legate index, we reuse its storage when
            # importing the Pandas series
            if isinstance(index, BaseIndex):
                tmp_sr = pandas.Series(
                    data=data,
                    dtype=dtype,
                    name=name,
                    copy=copy,
                )
                self._frame = from_pandas(tmp_sr, index=index)

            # Otherwise, we construct the complete Pandas series
            # and import it
            else:
                tmp_sr = pandas.Series(
                    data=data,
                    index=index,
                    dtype=dtype,
                    name=name,
                    copy=copy,
                )
                self._frame = from_pandas(tmp_sr)
            self._set_name(tmp_sr.name)

        assert len(self._frame._columns) == 1

    ###########
    # Utilities
    ###########

    @property
    def _is_series(self):
        return True

    def _get_columns(self):
        return (
            pandas.RangeIndex(1)
            if self.name is None
            else pandas.Index([self.name])
        )

    def _create_or_update_frame(self, new_frame, inplace):
        if inplace:
            self._frame = new_frame
            return None
        else:
            new_self = self.__ctor__(frame=new_frame, name=self.name)
            return new_self

    @property
    def __legate_data_interface__(self):
        return {
            "version": 1,
            "data": self._frame.to_legate_data(),
        }

    ################################
    # Attributes and underlying data
    ################################

    @copy_docstring(pandas.Series.axes)
    @property
    def axes(self):
        """
        Return a list of the row axis labels.
        """
        return [self.index]

    @property
    def values(self):
        """
        Return Series as ndarray or ndarray-like depending on the dtype.

        Returns
        -------
        out : numpy.ndarray or ndarray-like
        """
        return super(Series, self).to_numpy()

    @copy_docstring(pandas.Series.dtype)
    @property
    def dtype(self):
        return self._frame.dtypes[0]

    @copy_docstring(pandas.Series.shape)
    @property
    def shape(self):
        return (len(self._index),)

    @copy_docstring(pandas.Series.ndim)
    @property
    def ndim(self):
        return 1

    @copy_docstring(pandas.Series.hasnans)
    @property
    def hasnans(self):
        return self.isna().sum() > 0

    dtypes = dtype

    def _get_name(self):
        return self._name

    def _set_name(self, name):
        if not is_hashable(name):
            raise TypeError(
                f"{type(self).__name__}.name must be a hashable type"
            )
        self._name = name

    name = property(_get_name, _set_name, None, pandas.Series.name.__doc__)

    ############
    # Conversion
    ############

    def __array__(self, dtype=None):
        return self.to_numpy(dtype)

    def copy(self, deep=True):
        if deep:
            return self.__ctor__(frame=self._frame.copy(), name=self.name)
        return self.__ctor__(frame=self._frame, name=self.name)

    def __copy__(self, deep=True):
        return self.copy(deep=deep)

    def __repr__(self):
        return _repr_series(self)

    #####################
    # Indexing, iteration
    #####################

    @copy_docstring(pandas.Series.at)
    @property
    def at(self):
        from .indexing import _LocSeriesLocator

        return _LocSeriesLocator(self, is_at=True)

    @copy_docstring(pandas.Series.iat)
    @property
    def iat(self):
        from .indexing import _IlocSeriesLocator

        return _IlocSeriesLocator(self, is_at=True)

    @copy_docstring(pandas.Series.loc)
    @property
    def loc(self):
        from .indexing import _LocSeriesLocator

        return _LocSeriesLocator(self)

    @copy_docstring(pandas.Series.iloc)
    @property
    def iloc(self):
        from .indexing import _IlocSeriesLocator

        return _IlocSeriesLocator(self)

    @copy_docstring(pandas.Series.where)
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

    @copy_docstring(pandas.Series.mask)
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

    def __getattr__(self, key):
        try:
            return object.__getattribute__(self, key)
        except AttributeError as e:
            if hasattr(pandas.Series, key):
                raise err._unsupported_error(
                    f"Series.{key} is not yet implemented in Legate Pandas."
                )
            raise e

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.iloc[key]
        else:
            # TODO: This isn't quite right because in Pandas, 'key' can
            #       sometimes denote an absolute position in the series,
            #       in which case we should be using iloc instead.
            #       We use this logic for now for simplicity (and this won't
            #       cause any major issue as the majority of indices would
            #       be integers anyway and users can use iloc if they want
            #       different behavior). Note that cuDF has the same issue
            #       (GH #7622).
            return self.loc[key]

    def __setitem__(self, key, item):
        if isinstance(key, slice):
            self.iloc[key] = item
        else:
            # TODO: this has the same issue as __getitem__ (see above).
            self.loc[key] = item

    def __delitem__(self, key):
        if key not in self.keys():
            raise KeyError(key)
        self.drop(labels=key, inplace=True)

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

        rhs = rhs._create_or_update_frame(
            lhs._raw_index._align_partition(rhs._frame), False
        )

        if not lhs._raw_index.equals(rhs._raw_index):
            raise NotImplementedError(
                "Unaligend dataframes are not supported yet"
            )

        name = lhs.name if rhs._is_series and lhs.name == rhs.name else None

        lhs = Series(frame=lhs._frame, name=name)
        if rhs._is_series:
            rhs = Series(frame=rhs._frame, name=name)

        return lhs, rhs

    def _broadcast(self, columns):
        frame = self._frame.broadcast(len(columns))

        from .dataframe import DataFrame

        return DataFrame(frame=frame, columns=columns)

    @copy_docstring(pandas.Series.add)
    def add(self, other, level=None, fill_value=None, axis=0):
        return self._binary_op(
            "add", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.Series.floordiv)
    def floordiv(self, other, level=None, fill_value=None, axis=0):
        return self._binary_op(
            "floordiv", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.Series.mod)
    def mod(self, other, level=None, fill_value=None, axis=0):
        return self._binary_op(
            "mod", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.Series.mul)
    def mul(self, other, level=None, fill_value=None, axis=0):
        return self._binary_op(
            "mul", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.Series.pow)
    def pow(self, other, level=None, fill_value=None, axis=0):
        return self._binary_op(
            "pow", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.Series.sub)
    def sub(self, other, level=None, fill_value=None, axis=0):
        return self._binary_op(
            "sub", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.Series.truediv)
    def truediv(self, other, level=None, fill_value=None, axis=0):
        return self._binary_op(
            "truediv", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.Series.rfloordiv)
    def rfloordiv(self, other, level=None, fill_value=None, axis=0):
        return self._binary_op(
            "rfloordiv", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.Series.rmod)
    def rmod(self, other, level=None, fill_value=None, axis=0):
        return self._binary_op(
            "rmod", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.Series.rpow)
    def rpow(self, other, level=None, fill_value=None, axis=0):
        return self._binary_op(
            "rpow", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.Series.rsub)
    def rsub(self, other, level=None, fill_value=None, axis=0):
        return self._binary_op(
            "rsub", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.Series.rtruediv)
    def rtruediv(self, other, level=None, fill_value=None, axis=0):
        return self._binary_op(
            "rtruediv", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.Series.eq)
    def eq(self, other, level=None, fill_value=None, axis=0):
        return self._binary_op(
            "eq", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.Series.le)
    def le(self, other, level=None, fill_value=None, axis=0):
        return self._binary_op(
            "le", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.Series.lt)
    def lt(self, other, level=None, fill_value=None, axis=0):
        return self._binary_op(
            "lt", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.Series.ge)
    def ge(self, other, level=None, fill_value=None, axis=0):
        return self._binary_op(
            "ge", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.Series.gt)
    def gt(self, other, level=None, fill_value=None, axis=0):
        return self._binary_op(
            "gt", other, axis=axis, level=level, fill_value=fill_value
        )

    @copy_docstring(pandas.Series.ne)
    def ne(self, other, level=None, fill_value=None, axis=0):
        return self._binary_op(
            "ne", other, axis=axis, level=level, fill_value=fill_value
        )

    def __and__(self, other, level=None, fill_value=None, axis=0):
        return self._binary_op(
            "__and__", other, axis=axis, level=level, fill_value=fill_value
        )

    def __or__(self, other, level=None, fill_value=None, axis=0):
        return self._binary_op(
            "__or__", other, axis=axis, level=level, fill_value=fill_value
        )

    def __xor__(self, other, level=None, fill_value=None, axis=0):
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

    @copy_docstring(pandas.Series.groupby)
    def groupby(
        self,
        by=None,
        axis=0,
        level=None,
        sort=False,
        **kwargs,
    ):
        from .groupby import SeriesGroupBy

        return SeriesGroupBy(
            df=self,
            by=by,
            axis=axis,
            level=level,
            as_index=True,
            sort=sort,
            method=kwargs.get("method", "hash"),
        )

    ##################################
    # Computations / descriptive stats
    ##################################

    @copy_docstring(pandas.Series.groupby)
    def count(self, level=None):
        return super(Series, self).count(level=level)

    def __int__(self):
        return int(self.squeeze())

    #############################################
    # Reindexing / selection / label manipulation
    #############################################

    @copy_docstring(pandas.Series.equals)
    def equals(self, other):
        if not isinstance(other, type(self)):
            other = Series(other)
        return self._frame.equals(other._frame)

    @copy_docstring(pandas.Series.reset_index)
    def reset_index(self, level=None, drop=False, name=None, inplace=False):
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
        if inplace and len(frame._columns) > 1:
            raise TypeError(
                "Cannot reset_index inplace on a Series to create a DataFrame"
            )

        if drop:
            return self._create_or_update_frame(frame, inplace)

        if name is None:
            name = 0 if self.name is None else self.name
        names = self._raw_index._get_level_names(levels) + [name]
        columns = pandas.Index(names)

        from .dataframe import DataFrame

        return DataFrame(columns=columns, frame=frame)

    #######################
    # Missing data handling
    #######################

    #################################
    # Reshaping, sorting, transposing
    #################################

    @copy_docstring(pandas.Series.sort_values)
    def sort_values(
        self,
        axis=0,
        ascending=True,
        inplace=False,
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

        ascending = self._get_ascending(ascending, 1)
        new_frame = self._frame.sort_values(
            [0],
            axis,
            ascending,
            kind,
            na_position,
            ignore_index,
        )
        return self._create_or_update_frame(new_frame, inplace)

    @copy_docstring(pandas.Series.squeeze)
    def squeeze(self, axis=None):
        axis = self._get_axis_number(axis, 0)
        if len(self) == 1:
            return self.to_pandas().squeeze()
        else:
            return self

    ###########################################
    # Combining / comparing / joining / merging
    ###########################################

    #################################
    # Serialization / IO / conversion
    #################################

    @copy_docstring(pandas.Series.to_frame)
    def to_frame(self, name=None):
        from .dataframe import DataFrame

        new_self = self.copy()
        if name is not None:
            new_self.name = name
        return DataFrame(
            frame=new_self._frame, columns=new_self._get_columns()
        )

    @copy_docstring(pandas.Series.to_numpy)
    def to_numpy(
        self, dtype=None, copy=False, na_value=lib.no_default, **kwargs
    ):
        return self.to_pandas().to_numpy(dtype, copy, na_value, **kwargs)

    def to_pandas(self, schema_only=False):
        """
        Convert distributed Series into a Pandas Series

        Parameters
        ----------
        schema_only : Doesn't convert the data when True

        Returns
        -------
        out : pandas.Series
        """

        sr = self._frame.to_pandas(schema_only=schema_only).squeeze(axis=1)
        sr.name = self.name
        return sr

    ###########
    # Accessors
    ###########

    @copy_docstring(pandas.Series.cat)
    @property
    def cat(self):
        return CategoricalAccessor(self)

    @property
    def dt(self):
        """
        Accessor object for datetimelike properties of the Series values.
        """
        return DatetimeProperties(self)

    @copy_docstring(pandas.Series.str)
    @property
    def str(self):
        return StringMethods(self)

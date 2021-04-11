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
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_integer,
    is_scalar,
)

from legate.pandas.common import errors as err, util as util


class _NotFoundError(Exception):
    pass


def _compute_ndim(row_loc, col_loc=None):
    row_scalar = is_scalar(row_loc) or util.is_tuple(row_loc)
    col_scalar = is_scalar(col_loc) or util.is_tuple(col_loc)

    if row_scalar and col_scalar:
        ndim = 0
    elif row_scalar ^ col_scalar:
        ndim = 1
    else:
        ndim = 2

    return ndim


class _DataFrameLocator(object):
    def __init__(self, df, is_loc, is_at):
        self.df = df
        self.is_loc = is_loc
        self.is_at = is_at

    def construct_result(self, result, columns, out_ndim, row_scalar):
        if out_ndim > 0:
            result = self.df.__ctor__(frame=result, columns=columns)
            if out_ndim == 1:
                result = result.squeeze(axis=1)
        else:
            result = result.to_pandas().squeeze()

        if row_scalar and not is_scalar(result) and len(result) == 0:
            raise _NotFoundError()
        return result

    def update_columns(self, col_indexer, result):
        self.df._frame.update_columns(col_indexer, result)

    def _validate_locators(self, tup):
        if util.is_tuple(tup) and len(tup) >= 1:
            if len(tup) > 2:
                raise ValueError("Too many indexers")
            row_loc = tup[0]
            col_loc = tup[1] if len(tup) == 2 else slice(None)
        else:
            row_loc = tup
            col_loc = slice(None)

        if isinstance(row_loc, slice) and row_loc.step is not None:
            raise err._unsupported_error(
                "row slicer cannot have a step for now"
            )

        row_scalar = is_scalar(row_loc) or util.is_tuple(row_loc)
        col_scalar = is_scalar(col_loc) or util.is_tuple(col_loc)

        if self.is_at:
            if not util.is_tuple(tup) or len(tup) != 2:
                raise ValueError("Need two indexers")

            if self.is_loc:
                if not row_scalar or not col_scalar:
                    raise ValueError(
                        "At based indexing can only have scalar indexers"
                    )
            else:
                if not is_integer(row_loc) or not is_integer(col_loc):
                    raise ValueError(
                        "iAt based indexing can only have integer indexers"
                    )

        return (
            row_loc,
            [col_loc] if col_scalar else col_loc,
            row_scalar,
            col_scalar,
            _compute_ndim(row_loc, col_loc),
        )

    @staticmethod
    def _validate_lhs(frame):
        for dtype in frame._get_dtypes():
            if is_categorical_dtype(dtype):
                raise err._unsupported_error(
                    "Inplace updates to categorical columns are not supported"
                )

    def _compute_column_indexer(self, loc):
        # Use Pandas to generate an indexer
        if self.is_loc:
            return (
                pandas.Series(
                    range(len(self.df.columns)), index=self.df.columns
                )
                .loc[loc]
                .to_list()
            )
        else:
            return (
                pandas.RangeIndex(len(self.df.columns))
                .to_series()
                .iloc[loc]
                .index.to_list()
            )


class _LocDataframeLocator(_DataFrameLocator):
    def __init__(self, df, is_at=False):
        super().__init__(df, True, is_at)

    def __getitem__(self, key):
        (
            row_loc,
            col_loc,
            row_scalar,
            col_scalar,
            out_ndim,
        ) = self._validate_locators(key)

        col_indexer = self._compute_column_indexer(col_loc)
        projected = self.df._slice_columns(col_indexer)

        if row_scalar:
            index = projected._raw_index

            if index.nlevels == 1 and not is_scalar(row_loc):
                raise KeyError("row indexer must be a scalar")

            mask = index == row_loc

            result = projected._frame.select(mask)

            # If the frame has a multi-index, we need to check if it was
            # a partial match and handle the output accordingly (only to
            # make the output the same as Pandas' and for no other reason...)
            row_loc_tpl = util.to_tuple_if_scalar(row_loc)
            if index.nlevels > len(row_loc_tpl):
                # If this is a partial match, the output should not be
                # squeezed down to a scalar,
                out_ndim += 1
                # and the matched levels should be droped for some reason.
                result = result.droplevel(range(len(row_loc_tpl)))

        elif isinstance(row_loc, slice):
            if row_loc == slice(None):
                result = projected._frame
            else:
                if projected._raw_index.nlevels > 1:
                    raise err._unsupported_error(
                        "Slicing on a MultiIndex is not supported yet"
                    )

                result = projected._frame.slice_rows_by_slice(row_loc, True)

        else:
            row_loc = projected._ensure_valid_frame(row_loc)

            _, row_loc = projected._align_frame(row_loc, join="left", axis=0)

            if not row_loc._is_series:
                raise ValueError("indexer must be 1-dimensional")

            if not is_bool_dtype(row_loc.dtype):
                raise err._unsupported_error(
                    "only boolean indexers are supported now"
                )

            result = projected._frame.select(row_loc._frame)

        try:
            return super().construct_result(
                result, projected.columns, out_ndim, row_scalar
            )
        except _NotFoundError:
            raise KeyError(row_loc)

    def __setitem__(self, key, item):
        (
            row_loc,
            col_loc,
            row_scalar,
            col_scalar,
            _,
        ) = self._validate_locators(key)

        col_indexer = self._compute_column_indexer(col_loc)
        projected = self.df._slice_columns(col_indexer)

        self._validate_lhs(projected)

        if row_scalar:
            row_loc = projected._raw_index == row_loc

            index = projected._frame.slice_index_by_boolean_mask(row_loc)

            item = self._align_rhs(projected, index, item)

            result = projected._frame.scatter_by_boolean_mask(
                row_loc, index, item
            )

        elif isinstance(row_loc, slice):
            if row_loc == slice(None) and not is_scalar(item):
                index = projected._frame._index

                item = self._align_rhs(projected, index, item)

                result = item

            else:
                (index, bounds) = projected._frame.slice_index_by_slice(
                    row_loc, True
                )

                item = self._align_rhs(projected, index, item)

                result = projected._frame.scatter_by_slice(index, bounds, item)

        else:
            row_loc = projected._ensure_valid_frame(row_loc)
            _, row_loc = projected._align_frame(row_loc, join="left", axis=0)

            if not row_loc._is_series:
                raise ValueError("indexer must be 1-dimensional")

            if not is_bool_dtype(row_loc.dtype):
                raise err._unsupported_error(
                    "only boolean indexers are supported now"
                )

            row_loc = row_loc._frame

            index = projected._frame.slice_index_by_boolean_mask(row_loc)

            item = self._align_rhs(projected, index, item)

            result = projected._frame.scatter_by_boolean_mask(
                row_loc, index, item
            )

        self.update_columns(col_indexer, result)

    def _align_rhs(self, lhs, align_index, rhs):
        if not is_scalar(rhs):
            to_align = self.df.__ctor__(index=align_index, columns=lhs.columns)
            rhs = to_align._ensure_valid_frame(rhs)
            _, aligned = to_align._align_frame(
                rhs, join="left", broadcast_axis=1
            )
            # FIXME: For now we allow only aligned frames.
            if not (rhs._is_series or rhs.columns.equals(aligned.columns)):
                raise err._unsupported_error(
                    "Unaligned frames cannot be used for in-place updates"
                )
            rhs = aligned._frame

        return rhs


class _IlocDataframeLocator(_DataFrameLocator):
    def __init__(self, df, is_at=False):
        super().__init__(df, False, is_at)

    def __getitem__(self, key):
        (
            row_loc,
            col_loc,
            row_scalar,
            col_scalar,
            out_ndim,
        ) = self._validate_locators(key)

        col_indexer = self._compute_column_indexer(col_loc)
        projected = self.df._slice_columns(col_indexer)

        if row_scalar:
            result = projected._frame.read_at(row_loc)

        elif isinstance(row_loc, slice):
            if row_loc == slice(None):
                result = projected._frame
            else:
                result = projected._frame.slice_rows_by_slice(row_loc, False)

        else:
            row_loc = projected._ensure_valid_frame(row_loc)

            if not row_loc._is_series:
                raise ValueError("indexer must be 1-dimensional")

            if not is_bool_dtype(row_loc.dtype):
                raise err._unsupported_error(
                    "only boolean indexers are supported now"
                )

            # This may raise an exception if the indexer size doesn't match
            # with the index of the LHS.
            row_loc = row_loc._frame.update_legate_index(projected._raw_index)

            result = projected._frame.select(row_loc)

        try:
            return super().construct_result(
                result, projected.columns, out_ndim, row_scalar
            )
        except _NotFoundError:
            raise KeyError(row_loc)

    def __setitem__(self, key, item):
        (
            row_loc,
            col_loc,
            row_scalar,
            col_scalar,
            _,
        ) = self._validate_locators(key)

        col_indexer = self._compute_column_indexer(col_loc)
        projected = self.df._slice_columns(col_indexer)

        self._validate_lhs(projected)

        if row_scalar:
            (index, bounds) = projected._frame.slice_index_by_slice(
                slice(row_loc, row_loc + 1), False
            )

            # If the RHS is not a scalar, we need extra checks on the request
            if not is_scalar(item):
                # When both locators are scalars, they point to only one
                # location, and therefore the RHS must be a scalar
                if col_scalar:
                    raise ValueError("Value must be a scalar")

                # If we're updating multiple columns, the shape of RHS must
                # match with the LHS.
                else:
                    item = self._align_rhs(projected, index, item)

            result = projected._frame.scatter_by_slice(index, bounds, item)

        elif isinstance(row_loc, slice):
            if row_loc == slice(None):
                index = projected._frame._index

                item = self._align_rhs(projected, index, item)

                result = item

            else:
                (index, bounds) = projected._frame.slice_index_by_slice(
                    row_loc, False
                )

                item = self._align_rhs(projected, index, item)

                result = projected._frame.scatter_by_slice(index, bounds, item)

        else:
            row_loc = projected._ensure_valid_frame(row_loc)

            if not row_loc._is_series:
                raise ValueError("indexer must be 1-dimensional")

            if not is_bool_dtype(row_loc.dtype):
                raise err._unsupported_error(
                    "only boolean indexers are supported now"
                )

            # This may raise an exception if the indexer size doesn't match
            # with the index of the LHS.
            row_loc = row_loc._frame.update_legate_index(projected._raw_index)

            index = projected._frame.slice_index_by_boolean_mask(row_loc)

            item = self._align_rhs(projected, index, item)

            result = projected._frame.scatter_by_boolean_mask(
                row_loc, index, item
            )

        self.update_columns(col_indexer, result)

    def _align_rhs(self, lhs, align_index, rhs):
        if not is_scalar(rhs):
            # For iloc, we only check if the sizes match, which is performed by
            # this call.
            rhs._frame.update_legate_index(align_index)

            rhs = rhs._create_or_update_frame(
                align_index._align_partition(rhs._frame), False
            )

            _, aligned = lhs._align_frame(
                rhs, join="left", axis=1, broadcast_axis=1
            )
            # FIXME: For now we allow only aligned frames.
            if not (rhs._is_series or rhs.columns.equals(aligned.columns)):
                raise err._unsupported_error(
                    "Unaligned frames cannot be used for in-place updates"
                )
            rhs = aligned._frame

        return rhs


class _SeriesLocator(object):
    def __init__(self, sr, is_loc, is_at):
        self.sr = sr
        self.is_loc = is_loc
        self.is_at = is_at

    def construct_result(self, result, out_ndim, row_scalar):
        if out_ndim == 1:
            result = self.sr.__ctor__(frame=result, name=self.sr.name)
        else:
            assert out_ndim == 0
            result = result.to_pandas().squeeze()

        if row_scalar and not is_scalar(result) and len(result) == 0:
            raise _NotFoundError()

        return result

    def update_column(self, result):
        self.sr._frame.update_columns([0], result)

    def _validate_locator(self, row_loc):
        if util.is_tuple(row_loc):
            if len(row_loc) > 1:
                raise ValueError("Too many indexers")
            row_loc = row_loc[0]

        if isinstance(row_loc, slice) and row_loc.step is not None:
            raise err._unsupported_error(
                "row slicer cannot have a step for now"
            )

        row_scalar = is_scalar(row_loc) or util.is_tuple(row_loc)

        if self.is_at:
            if self.is_loc:
                if not row_scalar:
                    raise ValueError(
                        "At based indexing can only have scalar indexers"
                    )
            else:
                if not is_integer(row_loc):
                    raise ValueError(
                        "iAt based indexing can only have integer indexers"
                    )

        return (row_loc, row_scalar, _compute_ndim(row_loc))

    @staticmethod
    def _validate_lhs(series):
        if is_categorical_dtype(series.dtype):
            raise err._unsupported_error(
                "Inplace updates to categorical columns are not supported"
            )


class _LocSeriesLocator(_SeriesLocator):
    def __init__(self, df, is_at=False):
        super().__init__(df, True, is_at)

    def __getitem__(self, key):
        (row_loc, row_scalar, out_ndim) = self._validate_locator(key)

        sr = self.sr
        if row_scalar:
            index = sr._raw_index

            if index.nlevels == 1 and not is_scalar(row_loc):
                raise KeyError("row indexer must be a scalar")

            mask = index == row_loc

            result = sr._frame.select(mask)

            # If the frame has a multi-index, we need to check if it was
            # a partial match and handle the output accordingly (only to
            # make the output the same as Pandas' and for no other reason...)
            row_loc_tpl = util.to_tuple_if_scalar(row_loc)
            if index.nlevels > len(row_loc_tpl):
                # If this is a partial match, the output should not be
                # squeezed down to a scalar,
                out_ndim += 1
                # and the matched levels should be droped for some reason.
                result = result.droplevel(range(len(row_loc_tpl)))

        elif isinstance(row_loc, slice):
            if row_loc == slice(None):
                result = sr._frame
            else:
                result = sr._frame.slice_rows_by_slice(row_loc, True)

        else:
            row_loc = sr._ensure_valid_frame(row_loc)

            _, row_loc = sr._align_frame(row_loc, join="left", axis=0)

            if not is_bool_dtype(row_loc.dtype):
                raise err._unsupported_error(
                    "only boolean indexers are supported now"
                )

            result = sr._frame.select(row_loc._frame)

        try:
            return super().construct_result(result, out_ndim, row_scalar)
        except _NotFoundError:
            raise KeyError(row_loc)

    def __setitem__(self, key, item):
        (row_loc, row_scalar, _) = self._validate_locator(key)

        sr = self.sr

        self._validate_lhs(sr)

        if row_scalar:
            row_loc = sr._raw_index == row_loc

            index = sr._frame.slice_index_by_boolean_mask(row_loc)

            item = self._align_rhs(sr, index, item)

            result = sr._frame.scatter_by_boolean_mask(row_loc, index, item)

        elif isinstance(row_loc, slice):
            if row_loc == slice(None):
                index = sr._frame._index

                item = self._align_rhs(sr, index, item)

                result = item

            else:
                (index, bounds) = sr._frame.slice_index_by_slice(row_loc, True)

                item = self._align_rhs(sr, index, item)

                result = sr._frame.scatter_by_slice(index, bounds, item)

        else:
            row_loc = sr._ensure_valid_frame(row_loc)
            _, row_loc = sr._align_frame(row_loc, join="left", axis=0)

            if not row_loc._is_series:
                raise ValueError("indexer must be 1-dimensional")

            if not is_bool_dtype(row_loc.dtype):
                raise err._unsupported_error(
                    "only boolean indexers are supported now"
                )

            row_loc = row_loc._frame

            index = sr._frame.slice_index_by_boolean_mask(row_loc)

            item = self._align_rhs(sr, index, item)

            result = sr._frame.scatter_by_boolean_mask(row_loc, index, item)

        self.update_column(result)

    def _align_rhs(self, lhs, align_index, rhs):
        if not is_scalar(rhs):
            to_align = self.sr.__ctor__(index=align_index, name=lhs.name)
            _, aligned = to_align._align_frame(
                rhs, join="left", broadcast_axis=1
            )
            rhs = aligned._frame

        return rhs


class _IlocSeriesLocator(_SeriesLocator):
    def __init__(self, df, is_at=False):
        super().__init__(df, False, is_at)

    def __getitem__(self, key):
        (row_loc, row_scalar, out_ndim) = self._validate_locator(key)

        sr = self.sr
        if row_scalar:
            result = sr._frame.read_at(row_loc)

        elif isinstance(row_loc, slice):
            if row_loc == slice(None):
                result = sr._frame
            else:
                result = sr._frame.slice_rows_by_slice(row_loc, False)

        else:
            row_loc = sr._ensure_valid_frame(row_loc)

            if not row_loc._is_series:
                raise ValueError("indexer must be 1-dimensional")

            if not is_bool_dtype(row_loc.dtype):
                raise err._unsupported_error(
                    "only boolean indexers are supported now"
                )

            # This may raise an exception if the indexer size doesn't match
            # with the index of the LHS.
            row_loc = row_loc._frame.update_legate_index(sr._raw_index)

            result = sr._frame.select(row_loc)

        try:
            return super().construct_result(result, out_ndim, row_scalar)
        except _NotFoundError:
            raise KeyError(row_loc)

    def __setitem__(self, key, item):
        (row_loc, row_scalar, _) = self._validate_locator(key)

        sr = self.sr

        self._validate_lhs(sr)

        if row_scalar:
            if not is_scalar(item):
                raise ValueError("Value must be a scalar")

            result = sr._frame.write_at(row_loc, item)

        elif isinstance(row_loc, slice):
            if row_loc == slice(None):
                index = sr._frame._index

                item = self._align_rhs(sr, index, item)

                result = item

            else:
                (index, bounds) = sr._frame.slice_index_by_slice(
                    row_loc, False
                )

                item = self._align_rhs(sr, index, item)

                result = sr._frame.scatter_by_slice(index, bounds, item)

        else:
            row_loc = sr._ensure_valid_frame(row_loc)

            if not row_loc._is_series:
                raise ValueError("indexer must be 1-dimensional")

            if not is_bool_dtype(row_loc.dtype):
                raise err._unsupported_error(
                    "only boolean indexers are supported now"
                )

            # This may raise an exception if the indexer size doesn't match
            # with the index of the LHS.
            row_loc = row_loc._frame.update_legate_index(sr._raw_index)

            index = sr._frame.slice_index_by_boolean_mask(row_loc)

            item = self._align_rhs(sr, index, item)

            result = sr._frame.scatter_by_boolean_mask(row_loc, index, item)

        self.update_column(result)

    def _align_rhs(self, lhs, align_index, rhs):
        if not is_scalar(rhs):
            # For iloc, we only check if the sizes match, which is performed by
            # this call.
            rhs._frame.update_legate_index(align_index)
            rhs = align_index._align_partition(rhs._frame)

        return rhs

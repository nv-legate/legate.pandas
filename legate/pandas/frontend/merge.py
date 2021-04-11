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


from legate.pandas.common import errors as err, util as util


class Merge(object):
    def __init__(
        self,
        left,
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
        import pandas

        from .frame import Frame

        if not isinstance(right, (Frame, pandas.Series, pandas.DataFrame)):
            raise TypeError(
                f"Can only merge Series or DataFrame objects, "
                f"a {type(right)} was passed"
            )

        right = left._ensure_valid_frame(right)

        if right._is_series and right.name is None:
            raise ValueError("Cannot merge a Series without a name")

        # Checks overlap between column names
        if not util.is_tuple(suffixes) or len(suffixes) != 2:
            raise ValueError(f"Invalid suffixes: {suffixes}")
        if any(not isinstance(suffix, str) for suffix in suffixes):
            raise ValueError("Suffixes must be strings, but got {suffixes}")

        l_suffix, r_suffix = suffixes
        left_columns = left._get_columns()
        right_columns = right._get_columns()
        intersection = left_columns.intersection(right_columns)
        if len(intersection) != 0 and not (bool(l_suffix) or bool(r_suffix)):
            raise ValueError(
                f"columns overlap but no suffix specified: {intersection}"
            )

        # Perform Legate specific checks
        method = kwargs.get("method", "hash")

        if how not in (
            "inner",
            "left",
            "outer",
        ):
            raise err._unsupported_error("how", how)

        if how == "outer" and method == "broadcast":
            raise ValueError("Broadcast join cannot be used for outer join")

        if copy not in (True,):
            raise err._unsupported_error("copy", copy)

        if sort not in (False,):
            raise err._unsupported_error("sort", sort)

        self._left = left
        self._right = right

        self._how = how
        self._on = on

        self._left_on = left_on
        self._right_on = right_on

        self._left_index = left_index
        self._right_index = right_index

        self._left_columns = left_columns
        self._right_columns = right_columns

        self._sort = sort
        self._suffixes = suffixes
        self._copy = copy
        self._method = method

    def perform_merge(self):
        left = self._left._frame
        right = self._right._frame

        # FIXME: For now we will pass in column names to the core and
        # get the result column names back. We will eventually hoist
        # that logic out from the core and make it operate only on
        # the storage.
        frame, columns = left.merge(
            right,
            self._left_columns,
            self._right_columns,
            how=self._how,
            on=self._on,
            left_on=self._left_on,
            right_on=self._right_on,
            left_index=self._left_index,
            right_index=self._right_index,
            sort=self._sort,
            suffixes=self._suffixes,
            method=self._method,
        )

        from .dataframe import DataFrame

        return DataFrame(frame=frame, columns=columns)

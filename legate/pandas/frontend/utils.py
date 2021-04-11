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
from pandas.api.types import is_string_dtype

from legate.pandas.common import errors as err, util as util

from .dataframe import DataFrame
from .doc_utils import copy_docstring
from .frame import Frame
from .series import Series

###############
# Concatenation
###############

_CTORS = {
    pandas.DataFrame: DataFrame,
    pandas.Series: Series,
}


@copy_docstring(pandas.concat)
def concat(
    objs,
    axis=0,
    join="outer",
    ignore_index=False,
    keys=None,
    levels=None,
    names=None,
    verify_integrity=False,
    sort=False,
    copy=True,
):
    if isinstance(objs, (pandas.Series, pandas.DataFrame, Frame)):
        raise TypeError(
            "first argument must be an iterable of pandas "
            "objects, you passed an object of type "
            '"{name}"'.format(name=type(objs).__name__)
        )

    if join not in (
        "outer",
        "inner",
    ):
        raise ValueError(
            "Only can inner (intersect) or outer (union) join the other axis"
        )

    if levels is not None:
        raise err._unsupported_error("levels", levels)

    objs = [obj for obj in list(objs) if obj is not None]
    if len(objs) == 0:
        raise ValueError("No objects to concatenate")

    to_concat = []
    for obj in objs:
        if not isinstance(obj, (pandas.Series, pandas.DataFrame, Frame)):
            raise TypeError(
                f"cannot concatenate object of type '{type(obj)}'; "
                "only Series and DataFrame objs are valid "
            )
        to_concat.append(
            obj if isinstance(obj, Frame) else _CTORS[type(obj)](obj)
        )
    objs = to_concat

    if len(objs) == 1:
        return objs[0]

    axis = DataFrame._get_axis_number(axis)
    first = objs[0]
    others = objs[1:]
    frames = [obj._frame for obj in objs]
    all_series = all(obj._is_series for obj in objs)
    if axis == 1:
        # TODO: Here we need to join the indices and reindex the series
        #       using that joined index. Since we haven't implemented
        #       reindex, we will align the first series with the others
        #       just to make sure that they are aligned (_align_frame
        #       is currently doing nothing more than checking that
        #       the indices are the same).

        others = [first._align_frame(other, axis=0)[1] for other in others]

        columns = []
        num = 0
        for obj in objs:
            if obj._is_series:
                if obj.name is not None:
                    columns.append(obj.name)
                else:
                    columns.append(num)
                    num += 1
            else:
                columns.extend(obj.columns)
        columns = pandas.Index(columns)
        return DataFrame(
            columns=columns, frame=frames[0].concat(1, frames[1:])
        )

    else:
        others = [first._align_frame(other, axis=1)[1] for other in others]

        new_frame = frames[0].concat(0, frames[1:])

        columns = first._get_columns()
        if len(columns) == 1 and all_series:
            return Series(name=columns[0], frame=new_frame)
        else:
            return DataFrame(columns=columns, frame=new_frame)


#############
# to_datetime
#############


@copy_docstring(pandas.to_datetime)
def to_datetime(
    arg,
    errors="raise",
    dayfirst=False,
    yearfirst=False,
    utc=None,
    format=None,
    exact=True,
    unit=None,
    infer_datetime_format=False,
    origin="unix",
    cache=True,
):
    if not isinstance(arg, Frame):
        result = pandas.to_datetime(
            arg,
            errors=errors,
            dayfirst=dayfirst,
            yearfirst=yearfirst,
            utc=utc,
            format=format,
            exact=exact,
            unit=unit,
            infer_datetime_format=infer_datetime_format,
            origin=origin,
            cache=cache,
        )
        return util.sanitize_scalar(result)

    if not (arg._is_series and is_string_dtype(arg.dtype)):
        print(type(arg.dtype))
        raise err._unsupported_error("to_datetime handles only string columns")

    return arg.str.to_datetime(format)

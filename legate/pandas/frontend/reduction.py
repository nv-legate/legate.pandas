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


from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_dict_like,
    is_list_like,
    is_numeric_dtype,
)

from legate.pandas.common import errors as err, util as util

_SUPPORTED_AGGS = [
    "all",
    "any",
    "count",
    "max",
    "mean",
    "min",
    "prod",
    "size",
    "std",
    "sum",
    "var",
]

_NUMERIC_ONLY = {
    "all": True,
    "any": True,
    "count": False,
    "max": False,
    "mean": True,
    "min": False,
    "prod": True,
    "size": False,
    "std": True,
    "sum": True,
    "var": True,
}


def convert_agg_func(agg_func):
    if isinstance(agg_func, str):
        if agg_func not in _SUPPORTED_AGGS:
            raise err._unsupported_error(
                f"Unsupported aggregation method: {agg_func}"
            )
        return (agg_func, _NUMERIC_ONLY[agg_func])
    elif is_dict_like(agg_func):
        converted = {}
        for col, func in agg_func.items():
            funcs = util.to_list_if_scalar(convert_agg_func(func))
            converted[col] = funcs
        return converted
    elif is_list_like(agg_func):
        return [convert_agg_func(func) for func in agg_func]
    else:
        raise err._unsupported_error(
            f"Unsupported aggregation descriptor: {agg_func}"
        )


def incompatible_ops(descs, dtype):
    if is_categorical_dtype(dtype):
        return any(
            desc[0]
            not in (
                "size",
                "count",
            )
            for desc in descs
        )

    numeric_only = any(desc[1] for desc in descs)
    return numeric_only and not (
        is_numeric_dtype(dtype) or is_bool_dtype(dtype)
    )


def _maybe_convert_to_default(desc):
    op, numeric_only = desc
    if numeric_only is None:
        return (op, _NUMERIC_ONLY[op])
    else:
        return desc


def unary_reduction(df, ops, axis=0, skipna=True, level=None):
    if isinstance(ops, list):
        ops = [_maybe_convert_to_default(desc) for desc in ops]

    else:
        # TODO: We will hit this case once we add agg/aggregate
        assert False

    if axis != 0:
        raise err._unsupported_error("axis", axis)
    if skipna not in (
        True,
        None,
    ):
        raise err._unsupported_error("skipna", skipna)
    if level is not None:
        raise err._unsupported_error("level", level)

    columns = df._frame._columns

    indexer = []
    for idx, column in enumerate(columns):
        if incompatible_ops(ops, column.dtype.to_pandas()):
            continue
        indexer.append(idx)

    valid_columns = [columns[idx] for idx in indexer]
    ops = [desc[0] for desc in ops]

    if df._is_series:
        if len(valid_columns) == 0:
            raise TypeError(
                f"Cannot perform reduction '{ops[0]}' "
                f"with {columns[0].dtype} dtype"
            )
        result = valid_columns[0].unary_reduction(ops[0], skipna)
        return result.get_scalar().value

    else:
        frame = df._frame.replace_columns(valid_columns)
        columns = df.columns[indexer]
        new_frame = frame.unary_reduction(
            ops[0],
            columns,
            axis=axis,
            skipna=skipna,
            level=level,
        )

        if len(new_frame._columns) > 1:
            from .dataframe import DataFrame

            return DataFrame(frame=new_frame, columns=df.columns)

        else:
            from .series import Series

            return Series(frame=new_frame)

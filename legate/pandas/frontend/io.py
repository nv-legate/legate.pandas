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

import pandas
from pandas.api.types import (
    is_categorical_dtype,
    is_dict_like,
    is_integer,
    is_list_like,
)
from pandas.io.common import infer_compression

from legate.pandas.common import errors as err, types as ty, util as util
from legate.pandas.config import CompressionType
from legate.pandas.core import io

from .dataframe import DataFrame
from .doc_utils import copy_docstring


@copy_docstring(pandas.read_parquet)
def read_parquet(path, columns=None, **kwargs):
    return DataFrame(**io.read_parquet(path=path, columns=columns, **kwargs))


def _ensure_dtype(dtype):
    if is_categorical_dtype(dtype):
        return "category"
    return ty.to_legate_dtype(dtype)


def _extract_header_using_pandas(
    path,
    sep,
    header,
    names,
    dtype,
    true_values,
    false_values,
    skiprows,
    na_values,
    skip_blank_lines,
    parse_dates,
    compression,
    quotechar,
    quoting,
    doublequote,
    engine,
    peek_rows=3,
):
    df = pandas.read_csv(
        path,
        sep=sep,
        header=header,
        names=names,
        dtype=dtype,
        true_values=true_values,
        false_values=false_values,
        skiprows=skiprows,
        na_values=na_values,
        skip_blank_lines=skip_blank_lines,
        parse_dates=parse_dates,
        compression=compression,
        quotechar=quotechar,
        quoting=quoting,
        doublequote=doublequote,
        nrows=peek_rows,
    )

    dtypes = [_ensure_dtype(dtype) for dtype in df.dtypes]

    return df.columns, dtypes


def _get_indexer(columns, to_lookup, opt_name):
    indexer = []
    for val in to_lookup:
        if is_integer(val):
            indexer.append(val)
        elif isinstance(val, str):
            idxr = columns.get_indexer_for([val])
            if idxr[0] == -1:
                raise KeyError(val)
            indexer.append(idxr[0])
        else:
            raise ValueError(
                f"Unsupported value type {type(val)} for '{opt_name}'"
            )
    return indexer


def _check_string_list(values, opt_name):
    if values is None:
        return
    if isinstance(values, str):
        values = [values]
    if not is_list_like(values) or any(
        not isinstance(val, str) for val in values
    ):
        raise ValueError(f"'{opt_name}' must be a list of strings")


def _parse_compression(compression):
    if compression is None:
        return CompressionType.UNCOMPRESSED
    else:
        return CompressionType[compression.upper()]


@copy_docstring(pandas.read_csv)
def read_csv(
    filepath_or_buffer,
    sep=",",
    delimiter=None,
    header="infer",
    names=None,
    index_col=None,
    usecols=None,
    prefix=None,
    mangle_dupe_cols=True,
    dtype=None,
    true_values=None,
    false_values=None,
    skiprows=None,
    skipfooter=0,
    nrows=None,
    na_values=None,
    skip_blank_lines=True,
    parse_dates=False,
    compression="infer",
    quotechar='"',
    quoting=0,
    doublequote=True,
    verify_header=False,
    **kwargs,
    # TODO: Put back these options once we figure out how to support them
    #       with the Arrows CSV reader.
    # skipinitialspace=False,  # GPU only
    # keep_default_na=True,  # GPU only
    # na_filter=True,  # GPU only
    # dayfirst=False, # GPU only
    # thousands=None,  # GPU only
    # decimal=".",  # GPU only
    # lineterminator=None, # GPU only
    # comment=None,  # GPU only
    # delim_whitespace=False,  # GPU only
):

    # Checks on filepath_or_buffer
    paths = util.to_list_if_scalar(filepath_or_buffer)

    if any(not isinstance(path, str) for path in paths):
        raise err._unsupported_error(
            "'filepath_or_buffer' must be a string or a list of strings"
        )
    if len(paths) == 0:
        raise ValueError("'filepath_or_buffer' must be a non-empty list")

    for path in paths:
        if not os.path.exists(path):
            raise ValueError(f"{path} does not exist")

    if not isinstance(compression, str):
        raise err._unsupported_error("compression", compression)
    compressions = [
        _parse_compression(infer_compression(path, compression))
        for path in paths
    ]

    # Checks on sep and delimiter
    if sep is None and delimiter is None:
        raise ValueError("at least one of 'sep' or 'delimiter' must be given")
    sep = delimiter if delimiter is not None else sep
    if len(sep) > 1:
        raise ValueError("'sep' must be a 1-character string")

    # Checks on sep and delimiter
    if header == "infer":
        header = 0 if names is None else None

    if header not in (
        0,
        None,
    ):
        raise err._unsupported_error("header", header)

    # Checks on skiprows, kipfooter, and nrows
    skiprows = 0 if skiprows is None else skiprows
    if not is_integer(skiprows):
        raise ValueError("'skiprows' must be an integer")
    if not is_integer(skipfooter):
        raise ValueError("'skipfooter' must be an integer")
    if not (nrows is None or is_integer(nrows)):
        raise ValueError("'nrows' must be None or an integer")

    # If either column names or dtype is missing, infer them by parsing
    # the first few of lines using Pandas
    # FIXME: We should use cuDF for this
    if names is None or dtype is None:
        engine = ("python" if skipfooter > 0 else "c",)
        column_names, dtypes = _extract_header_using_pandas(
            paths[0],
            sep,
            header,
            names,
            dtype,
            true_values,
            false_values,
            skiprows,
            na_values,
            skip_blank_lines,
            parse_dates,
            compression,
            quotechar,
            quoting,
            doublequote,
            engine,
            peek_rows=3,
        )
        if verify_header:
            for path in paths[1:]:
                result = _extract_header_using_pandas(
                    path,
                    sep,
                    header,
                    names,
                    dtype,
                    true_values,
                    false_values,
                    skiprows,
                    na_values,
                    skip_blank_lines,
                    parse_dates,
                    compression,
                    quotechar,
                    quoting,
                    doublequote,
                    engine,
                    peek_rows=3,
                )
                if not column_names.equals(result[0]):
                    raise ValueError(
                        f"{paths[0]} and {path} have different headers"
                    )

    else:
        column_names = pandas.Index(names)

        if is_dict_like(dtype):
            dtypes = []
            for name in names:
                if name not in dtype:
                    raise ValueError(f"'dtype' has no entry for '{name}'")
                dtypes.append(_ensure_dtype(dtype[name]))
        elif is_list_like(dtype):
            raise err._unsupported_error(
                "'dtype' must be a string, a dtype, or a dictionary"
            )
        else:
            dtype = _ensure_dtype(dtype)
            dtypes = [dtype] * len(names)

    if column_names.has_duplicates:
        raise ValueError("Header must not have any duplicates")

    # Checks on unsupported options
    if prefix is not None:
        raise err._unsupported_error("prefix", prefix)
    if mangle_dupe_cols not in (True,):
        raise err._unsupported_error("mangle_dupe_cols", mangle_dupe_cols)

    # If there was a header in the file, we should skip that line as well
    if header == 0:
        skiprows += 1

    # Checks on parse_dates
    _ERR_MSG_PARSE_DATES = (
        "'parse_dates' must be a list of integers or strings for now"
    )

    if is_dict_like(parse_dates):
        raise err._unsupported_error(_ERR_MSG_PARSE_DATES)

    parse_dates = parse_dates if parse_dates is not False else []
    if not is_list_like(parse_dates):
        raise err._unsupported_error(_ERR_MSG_PARSE_DATES)

    date_cols = _get_indexer(column_names, parse_dates, "parse_dates")

    # Override dtypes for the datetime columns
    for idx in date_cols:
        dtypes[idx] = ty.ts_ns

    # If a column is given a datetime dtype but not added to the parse_dates,
    # we should record it
    for idx, dtype in enumerate(dtypes):
        if idx not in parse_dates:
            parse_dates.append(idx)

    # Checks on quoting
    if quoting != 0:
        raise err._unsupported_error("quoting", quoting)
    if len(quotechar) > 1:
        raise ValueError("'quotechar' must be a 1-character string")

    # Checks on index_col
    index_col = None if index_col is False else index_col
    if index_col is not None:
        if is_integer(index_col) or isinstance(index_col, str):
            index_col = [index_col]
        if not is_list_like(index_col):
            raise err._unsupported_error("index_col", index_col)
        index_col = _get_indexer(column_names, index_col, "index_col")

    # Checks on true_values, false_values, and na_values
    _check_string_list(true_values, "true_values")
    _check_string_list(false_values, "false_values")
    _check_string_list(na_values, "na_values")

    # Checks on nrows
    if skipfooter != 0 and nrows is not None:
        raise ValueError("'skipfooter' not supported with 'nrows'")

    df = DataFrame(
        frame=io.read_csv(
            paths,
            sep=sep,
            usecols=usecols,
            dtypes=dtypes,
            true_values=true_values,
            false_values=false_values,
            skiprows=skiprows,
            skipfooter=skipfooter,
            nrows=nrows,
            na_values=na_values,
            skip_blank_lines=skip_blank_lines,
            date_cols=date_cols,
            compressions=compressions,
            quotechar=quotechar,
            quoting=quoting,
            doublequote=doublequote,
        ),
        columns=column_names,
    )

    if index_col is not None:
        df = df.set_index(column_names[index_col])
        # Make sure we reset the names for unnamed indices
        names = df._raw_index.names
        names = [
            None if name.startswith("Unnamed") else name for name in names
        ]
        df._raw_index.names = names

    return df


@copy_docstring(pandas.read_table)
def read_table(
    filepath_or_buffer,
    sep="\t",
    delimiter=None,
    header="infer",
    names=None,
    index_col=None,
    usecols=None,
    squeeze=False,
    prefix=None,
    mangle_dupe_cols=True,
    dtype=None,
    true_values=None,
    false_values=None,
    na_values=None,
    skip_blank_lines=True,
    parse_dates=False,
    compression="infer",
    quotechar='"',
    quoting=0,
    skipfooter=0,
    skiprows=None,
    nrows=None,
    doublequote=True,
    **kwargs,
    # TODO: Put back these options once we figure out how to support them
    #       with the Arrows CSV reader.
    # skipinitialspace=False,  # GPU only
    # keep_default_na=True,  # GPU only
    # na_filter=True,  # GPU only
    # dayfirst=False, # GPU only
    # thousands=None,  # GPU only
    # decimal=".",  # GPU only
    # lineterminator=None, # GPU only
    # comment=None,  # GPU only
    # delim_whitespace=False,  # GPU only
):
    return read_csv(
        filepath_or_buffer,
        sep=sep,
        delimiter=delimiter,
        header=header,
        names=names,
        index_col=index_col,
        usecols=usecols,
        prefix=prefix,
        mangle_dupe_cols=mangle_dupe_cols,
        dtype=dtype,
        true_values=true_values,
        false_values=false_values,
        skiprows=skiprows,
        skipfooter=skipfooter,
        nrows=nrows,
        na_values=na_values,
        skip_blank_lines=skip_blank_lines,
        parse_dates=parse_dates,
        compression=compression,
        quotechar=quotechar,
        quoting=quoting,
        doublequote=doublequote,
        **kwargs,
        # skipinitialspace=skipinitialspace,
        # keep_default_na=keep_default_na,
        # na_filter=na_filter,
        # dayfirst=dayfirst,
        # thousands=thousands,
        # decimal=decimal,
        # lineterminator=lineterminator,
        # comment=comment,
        # delim_whitespace=delim_whitespace,
    )

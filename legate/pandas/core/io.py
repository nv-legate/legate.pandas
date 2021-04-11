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
import shutil

from pandas.api.types import is_list_like

from legate.pandas.common import types as ty, util as util
from legate.pandas.config import CompressionType, OpCode

from .index import create_index_from_columns, create_range_index
from .pattern import Map
from .table import Table


def read_parquet(path, columns, **kwargs):
    from legate.core import Rect

    from .runtime import _runtime as rt

    path = util.to_list_if_scalar(path)

    if len(path) == 1 and os.path.isdir(path[0]):
        from pyarrow.parquet import ParquetDataset

        ds = ParquetDataset(path)
        path = [piece.path for piece in ds.pieces]
    else:
        from pyarrow.parquet import ParquetFile

        ds = ParquetFile(path[0])
        if rt.debug:
            assert all(ParquetFile(p).schema == ds.schema for p in path)

    dedup_names = set()
    for name in ds.schema.names:
        if name in dedup_names:
            raise ValueError(
                "Duplicate column names in schema are not supported."
            )
        dedup_names.add(name)

    schema = ds.schema.to_arrow_schema()
    index_descs = []
    index_materialized = False
    if str.encode("pandas") in ds.metadata.metadata:
        import json

        pandas_metadata = json.loads(
            ds.metadata.metadata[str.encode("pandas")]
        )
        index_descs = pandas_metadata["index_columns"]
        index_materialized = len(index_descs) > 0 and all(
            isinstance(desc, str) for desc in index_descs
        )

    if columns is None:
        column_names = schema.names
    elif index_materialized:
        column_names = columns + index_descs
    else:
        column_names = columns

    for name in column_names:
        if name not in dedup_names:
            raise ValueError("Field named %s not found in the schema." % name)
    schema = [schema.field(name) for name in column_names]
    del columns

    storage = rt.create_output_storage()
    offsets_storage = None

    columns = []
    for column_info in schema:
        dtype = ty.to_legate_dtype(column_info.type)
        column = storage.create_column(dtype)
        if ty.is_string_dtype(dtype):
            if offsets_storage is None:
                offsets_storage = rt.create_output_storage()
            offsets_column = offsets_storage.create_column(
                ty.int32, nullable=False
            )
            chars_storage = rt.create_output_storage()
            char_column = chars_storage.create_column(ty.int8, nullable=False)
            column.add_child(offsets_column)
            column.add_child(char_column)
            column = column.as_string_column()
        columns.append(column)

    plan = Map(rt, OpCode.READ_PARQUET)
    plan.add_scalar_arg(len(path), ty.uint32)
    for f in path:
        plan.add_scalar_arg(f, ty.string)
    plan.add_scalar_arg(len(column_names), ty.uint32)
    for name in column_names:
        plan.add_scalar_arg(name, ty.string)
    plan.add_scalar_arg(len(columns), ty.uint32)
    for column in columns:
        column.add_to_plan_output_only(plan)
    counts = plan.execute(Rect([rt.num_pieces]))
    storage = plan.promote_output_storage(storage)
    rt.register_external_weighted_partition(storage.default_ipart, counts)
    del plan

    size = counts.cast(ty.int64).sum()

    if index_materialized:
        to_filter = set(index_descs)

        index_columns = []
        value_columns = []
        value_column_names = []
        for idx, name in enumerate(column_names):
            if name in to_filter:
                index_columns.append(columns[idx])
            else:
                value_columns.append(columns[idx])
                value_column_names.append(column_names[idx])

        sanitized_names = [
            None if name == f"__index_level_{level}__" else name
            for level, name in enumerate(index_descs)
        ]
        index = create_index_from_columns(index_columns, size, sanitized_names)
    else:
        value_columns = columns
        value_column_names = column_names
        if len(index_descs) > 0:
            assert len(index_descs) == 1
            index_desc = index_descs[0]
            name = index_desc["name"]
            start = rt.create_future(index_desc["start"], ty.int64)
            stop = rt.create_future(index_desc["stop"], ty.int64)
            step = rt.create_future(index_desc["step"], ty.int64)
            index = create_range_index(storage, size, name, start, stop, step)
        else:
            index = create_range_index(storage, size)

    from pandas import Index

    return {
        "frame": Table(rt, index, value_columns),
        "columns": Index(value_column_names),
    }


def _may_add_to_plan(plan, value, value_dtype):
    plan.add_scalar_arg(value is not None, ty.bool)
    if value is None:
        return
    if is_list_like(value):
        plan.add_scalar_arg(len(value), ty.uint32)
        for val in value:
            plan.add_scalar_arg(val, value_dtype)
    else:
        plan.add_scalar_arg(value, value_dtype)


def _uncompress_files(paths, compressions):
    new_paths = []
    to_remove = []

    for path, compression in zip(paths, compressions):
        if compression == CompressionType.UNCOMPRESSED:
            new_paths.append(path)
            continue

        import tempfile

        out = os.path.join(
            tempfile.gettempdir(),
            f"_lg_uncompressed_{os.path.basename(path).replace('.gz', '')}",
        )
        new_paths.append(out)
        to_remove.append(out)

        if compression == CompressionType.GZIP:
            import gzip as decompress

        elif compression == CompressionType.BZ2:
            import bz2 as decompress

        else:
            from legate.pandas.common import errors as err

            raise err._unsupported_error(
                f"unsupported compression method '{compression.name.lower()}'"
            )

        with open(out, "wb") as f_out:
            with decompress.open(path, "rb") as f_in:
                shutil.copyfileobj(f_in, f_out)
    return new_paths, [CompressionType.UNCOMPRESSED] * len(paths), to_remove


def read_csv(
    paths,
    sep=None,
    usecols=None,
    dtypes=None,
    true_values=None,
    false_values=None,
    skiprows=0,
    skipfooter=0,
    nrows=None,
    na_values=None,
    skip_blank_lines=True,
    date_cols=False,
    compressions=None,
    quotechar='"',
    quoting=0,
    doublequote=True,
):
    from legate.core import Rect

    from .runtime import _runtime as rt

    storage = rt.create_output_storage()
    offsets_storage = None

    # Override the dtype for category columns, as they are not directly
    # handled by the CSV reader
    storage_dtypes = [
        ty.string if dtype == "category" else dtype for dtype in dtypes
    ]
    columns = [storage.create_column(dtype) for dtype in storage_dtypes]
    for column in columns:
        if ty.is_string_dtype(column.dtype):
            if offsets_storage is None:
                offsets_storage = rt.create_output_storage()
            offsets_column = offsets_storage.create_column(
                ty.int32, nullable=False
            )
            chars_storage = rt.create_output_storage()
            char_column = chars_storage.create_column(ty.int8, nullable=False)
            column.add_child(offsets_column)
            column.add_child(char_column)
    columns = [
        column.as_string_column()
        if ty.is_string_dtype(column.dtype)
        else column
        for column in columns
    ]

    # TODO: Since Arrow doesn't support in-flight decompression, we decompress
    #       any compressed files before tossing them to the reader.
    to_remove = []
    if not rt.has_gpus:
        paths, compressions, to_remove = _uncompress_files(paths, compressions)

    plan = Map(rt, OpCode.READ_CSV)
    plan.add_scalar_arg(len(paths), ty.uint32)
    for path in paths:
        plan.add_scalar_arg(path, ty.string)
    plan.add_scalar_arg(len(compressions), ty.uint32)
    for compression in compressions:
        plan.add_scalar_arg(compression.value, ty.int32)
    plan.add_scalar_arg(sep, ty.string)
    plan.add_scalar_arg(skiprows, ty.int32)
    plan.add_scalar_arg(skipfooter, ty.int32)
    _may_add_to_plan(plan, nrows, ty.int32)
    plan.add_scalar_arg(quotechar, ty.string)
    plan.add_scalar_arg(doublequote, ty.bool)
    plan.add_scalar_arg(skip_blank_lines, ty.bool)
    _may_add_to_plan(plan, true_values, ty.string)
    _may_add_to_plan(plan, false_values, ty.string)
    _may_add_to_plan(plan, na_values, ty.string)
    plan.add_scalar_arg(len(columns), ty.uint32)
    for column in columns:
        column.add_to_plan_output_only(plan)
    plan.add_scalar_arg(len(date_cols), ty.uint32)
    for idx in date_cols:
        plan.add_scalar_arg(idx, ty.int32)
    counts = plan.execute(Rect([rt.num_pieces]))
    storage = plan.promote_output_storage(storage)
    rt.register_external_weighted_partition(storage.default_ipart, counts)
    del plan

    columns = [
        column.to_category_column() if dtype == "category" else column
        for column, dtype in zip(columns, dtypes)
    ]

    size = counts.cast(ty.int64).sum()
    index = create_range_index(storage, size)

    if len(to_remove) > 0:
        counts.wait()
        for path in to_remove:
            os.remove(path)

    return Table(rt, index, columns)

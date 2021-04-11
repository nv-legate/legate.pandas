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

import numpy as np
import pandas as pd
import pyarrow as pa
from numpy import nan
from pandas import NaT
from pandas.core.dtypes import common as pandas_dtype
from pyarrow import types as pyarrow_dtype

from legate.pandas.common import errors as err
from legate.pandas.config import AggregationCode
from legate.pandas.library import c_header


class LegatePandasDtype(object):
    def __init__(self, name, size):
        self.name = name
        self.itemsize = size
        self.storage_dtype = self

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return (
            self is other
            or type(self) == type(other)
            and self.itemsize == other.itemsize
        )

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return str(self)

    def to_pandas(self):
        # TODO: For now we return a numpy type.
        #       We will return a nullable pandas dtype.
        return np.dtype(self.name)

    def to_arrow(self):
        return getattr(pa, self.name)()


class BoolDtype(LegatePandasDtype):
    def __init__(self):
        super(BoolDtype, self).__init__("bool", 1)


class IntDtype(LegatePandasDtype):
    _FORMAT = "int%d"

    def __init__(self, size):
        super(IntDtype, self).__init__(self._FORMAT % (size * 8), size)


class UIntDtype(LegatePandasDtype):
    _FORMAT = "uint%d"

    def __init__(self, size):
        super(UIntDtype, self).__init__(self._FORMAT % (size * 8), size)


class FloatDtype(LegatePandasDtype):
    _FORMAT = "float%d"

    def __init__(self, size):
        super(FloatDtype, self).__init__(self._FORMAT % (size * 8), size)


class RangeDtype(LegatePandasDtype):
    def __init__(self):
        super(RangeDtype, self).__init__("range", 16)

    def to_arrow(self):
        raise ValueError(f"{type(self)} can't be converted to an arrow type")


bool = BoolDtype()

int8 = IntDtype(1)
int16 = IntDtype(2)
int32 = IntDtype(4)
int64 = IntDtype(8)

uint8 = UIntDtype(1)
uint16 = UIntDtype(2)
uint32 = UIntDtype(4)
uint64 = UIntDtype(8)

float32 = FloatDtype(4)
float64 = FloatDtype(8)

range64 = RangeDtype()


class TimestampDtype(LegatePandasDtype):
    _FORMAT = "datetime64[%s]"

    def __init__(self, kind):
        super(TimestampDtype, self).__init__(self._FORMAT % kind, 8)
        self.storage_dtype = int64

    def to_arrow(self):
        return pa.date64()


class StringDtype(LegatePandasDtype):
    def __init__(self):
        super(StringDtype, self).__init__("string", 0)
        self.storage_dtype = range64

    def to_pandas(self):
        return pd.StringDtype()

    def to_arrow(self):
        return pa.string()


class CategoricalDtype(LegatePandasDtype):
    def __init__(self, categories_column, ordered=False):
        super(CategoricalDtype, self).__init__("category", 0)
        self.categories_column = categories_column
        self.ordered = ordered

        self._categories = None

    def encode(self, category, unwrap=False, can_fail=False):
        code = self.categories_column.as_string_column().encode_category(
            category, can_fail
        )
        if unwrap:
            return code.get_value()
        else:
            return code

    @property
    def categories(self):
        if self._categories is None:
            self._categories = self.categories_column.to_pandas()
        return self._categories

    def __eq__(self, other):
        if self is other or other == "category":
            return True
        elif not isinstance(other, CategoricalDtype):
            return False
        elif self.ordered != other.ordered:
            return False
        elif self.ordered:
            return self.categories.equals(other.categories)
        else:
            return set(self.categories) == set(other.categories)

    def _compare_categories(self, other):
        return self.categories_column.equals(other.categories_column, True)

    def to_pandas(self):
        return pd.CategoricalDtype(self.categories, self.ordered)

    def to_arrow(self):
        raise ValueError(f"{type(self)} can't be converted to an arrow type")

    @classmethod
    def from_pandas(cls, runtime, dtype):
        if dtype.categories.dtype != object:
            raise err._unsupported_error("Categories must be strings for now")
        categories_storage = runtime.create_storage(len(dtype.categories))
        categories_column = runtime._create_string_column_from_pandas(
            categories_storage,
            dtype.categories,
            num_pieces=1,
        ).as_replicated_column()

        return cls(categories_column, dtype.ordered)


ts_ns = TimestampDtype("ns")

string = StringDtype()

_DTYPE_MAPPING = {
    "bool": bool,
    "int8": int8,
    "int16": int16,
    "int32": int32,
    "int": int64,
    "int64": int64,
    "uint8": uint8,
    "uint16": uint16,
    "uint32": uint32,
    "uint": uint64,
    "uint64": uint64,
    "float32": float32,
    "float64": float64,
    "float": float64,
    "double": float64,
    "date": ts_ns,
    "datetime": ts_ns,
    "datetime64": ts_ns,
    "datetime64[ns]": ts_ns,
    "str": string,
    "string": string,
    # FIXME: All objects are treated as strings for now, as they are
    #        the only kind of objects we support.
    "object": string,
}

_CTYPE_MAPPING = {
    bool: c_header.BOOL_PT,
    int8: c_header.INT8_PT,
    int16: c_header.INT16_PT,
    int32: c_header.INT32_PT,
    int64: c_header.INT64_PT,
    uint8: c_header.UINT8_PT,
    uint16: c_header.UINT16_PT,
    uint32: c_header.UINT32_PT,
    uint64: c_header.UINT64_PT,
    float32: c_header.FLOAT_PT,
    float64: c_header.DOUBLE_PT,
    range64: c_header.RANGE_PT,
    ts_ns: c_header.TS_NS_PT,
    string: c_header.STRING_PT,
}

_CTYPE_TO_DTYPE = {
    c_header.BOOL_PT: bool,
    c_header.INT8_PT: int8,
    c_header.INT16_PT: int16,
    c_header.INT32_PT: int32,
    c_header.INT64_PT: int64,
    c_header.UINT8_PT: uint8,
    c_header.UINT16_PT: uint16,
    c_header.UINT32_PT: uint32,
    c_header.UINT64_PT: uint64,
    c_header.FLOAT_PT: float32,
    c_header.DOUBLE_PT: float64,
    c_header.TS_NS_PT: ts_ns,
    c_header.STRING_PT: string,
}

_DTYPE_TO_FORMAT = {
    bool: "?",
    int8: "b",
    int16: "h",
    int32: "i",
    int64: "q",
    uint8: "B",
    uint16: "H",
    uint32: "I",
    uint64: "Q",
    float32: "f",
    float64: "d",
    ts_ns: "q",
}

_OP_NAME_TO_AGG_CODE = {
    "sum": AggregationCode.SUM,
    "min": AggregationCode.MIN,
    "max": AggregationCode.MAX,
    "count": AggregationCode.COUNT,
    "prod": AggregationCode.PROD,
    "mean": AggregationCode.MEAN,
    "var": AggregationCode.VAR,
    "std": AggregationCode.STD,
    "size": AggregationCode.SIZE,
    "any": AggregationCode.ANY,
    "all": AggregationCode.ALL,
    "sqsum": AggregationCode.SQSUM,
}


def is_legate_pandas_dtype(dtype):
    return isinstance(dtype, LegatePandasDtype)


def is_string_dtype(dtype):
    return isinstance(dtype, StringDtype)


def is_categorical_dtype(dtype):
    return isinstance(dtype, CategoricalDtype)


def is_timestamp_dtype(dtype):
    return isinstance(dtype, TimestampDtype)


def is_signed_int_dtype(dtype):
    return isinstance(dtype, IntDtype)


def is_unsigned_int_dtype(dtype):
    return isinstance(dtype, UIntDtype)


def is_integer_dtype(dtype):
    return isinstance(dtype, (IntDtype, UIntDtype))


def is_float_dtype(dtype):
    return isinstance(dtype, FloatDtype)


def is_numeric_dtype(dtype):
    return isinstance(dtype, (IntDtype, UIntDtype, FloatDtype))


def is_primitive_dtype(dtype):
    return isinstance(dtype, (IntDtype, UIntDtype, FloatDtype, TimestampDtype))


def to_legate_dtype(dtype):
    if type(dtype) == str:
        if dtype not in _DTYPE_MAPPING:
            raise ValueError(f"invalid dtype {dtype}")
        return _DTYPE_MAPPING[dtype]
    elif isinstance(dtype, np.dtype):
        if dtype.name not in _DTYPE_MAPPING:
            raise ValueError(f"unsupported dtype {dtype}")
        return _DTYPE_MAPPING[dtype.name]
    elif isinstance(dtype, pa.DataType):
        if pyarrow_dtype.is_string(dtype):
            return string
        else:
            return to_legate_dtype(dtype.to_pandas_dtype())
    elif pandas_dtype.is_bool_dtype(dtype):
        return bool
    elif pandas_dtype.is_string_dtype(dtype):
        return string
    else:
        try:
            return to_legate_dtype(np.dtype(dtype))
        except TypeError:
            raise TypeError("Unsupported dtype: %s " % str(dtype))


def code_to_dtype(code):
    return _CTYPE_TO_DTYPE[code]


def to_format_string(dtype):
    # TODO: We should make this a property of the type
    return _DTYPE_TO_FORMAT[dtype]


def encode_dtype(dtype):
    if is_categorical_dtype(dtype):
        return c_header.CAT32_PT
    return _CTYPE_MAPPING[dtype]


def is_range_dtype(dtype):
    return isinstance(dtype, RangeDtype)


def find_common_dtype(dtype1, dtype2):
    if is_categorical_dtype(dtype1) and is_categorical_dtype(dtype2):
        from legate.pandas.common import errors as err

        raise err._unsupported_error(
            "categorical dtypes are not supported yet"
        )

    if dtype1 == dtype2:
        return dtype1
    else:
        return to_legate_dtype(
            np.find_common_type([dtype1.to_pandas(), dtype2.to_pandas()], [])
        )


def ensure_valid_index_dtype(dtype):
    if is_signed_int_dtype(dtype):
        return int64
    elif is_unsigned_int_dtype(dtype):
        return uint64
    elif is_float_dtype(dtype):
        return float64
    else:
        return dtype


def promote_dtype(dtype):
    if is_categorical_dtype(dtype) or is_timestamp_dtype(dtype):
        return dtype
    elif not is_float_dtype(dtype):
        return float64
    else:
        return dtype


_NULLABLE_DTYPES = set(
    [FloatDtype, TimestampDtype, CategoricalDtype, StringDtype]
)


def is_nullable_dtype(dtype):
    return type(dtype) in _NULLABLE_DTYPES


def null_value(pandas_dtype):
    if is_categorical_dtype(pandas_dtype):
        return np.iinfo(np.uint32).max
    elif is_timestamp_dtype(pandas_dtype):
        return int(NaT)
    else:
        return nan


def get_aggregation_op_id(op):
    return _OP_NAME_TO_AGG_CODE[op]


# This function queries the result type for a binary operator
# and also checks the typing of the operator as a byproduct.
def get_binop_result_type(op, dtype1, dtype2):
    # FIXME: When dtype1 or dtype2 is a categorical dtype, the to_pandas call
    #        will inline map the region field storing categories, which is a
    #        blocking operation and should be avoided. We should rather
    #        compute the result dtype ourselves. For now, we depend on Pandas,
    #        as it also performs type checking.
    dtype = getattr(pd.Series, op)(
        pd.Series([], dtype=dtype1.to_pandas()),
        pd.Series([], dtype=dtype2.to_pandas()),
    ).dtype
    return to_legate_dtype(dtype)


def get_reduction_result_type(op, dtype):
    if type(op) == str:
        op = get_aggregation_op_id(op)

    if op in (
        AggregationCode.COUNT,
        AggregationCode.SIZE,
    ):
        return int32
    elif op in (
        AggregationCode.MEAN,
        AggregationCode.VAR,
        AggregationCode.STD,
    ):
        return float64
    elif op in (
        AggregationCode.ANY,
        AggregationCode.ALL,
    ):
        return bool
    else:
        return dtype


def get_dt_field_type(dtype, field):
    # XXX: cuDF uses 16-bit integers for datetime fields
    return int16


def infer_dtype(val):
    dtype = np.array([val]).dtype
    if dtype.type == np.str_:
        return string
    elif dtype == "M8[ns]":
        return ts_ns
    else:
        return to_legate_dtype(dtype)

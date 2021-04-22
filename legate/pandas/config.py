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

from __future__ import absolute_import, division, print_function

from enum import IntEnum, unique

from legate.core import legion

from legate.pandas.library import c_header

PANDAS_FIELD_ID_BASE = 100


pandas_reduction_op_offsets = {
    "sum": legion.LEGION_REDOP_KIND_SUM,
    "prod": legion.LEGION_REDOP_KIND_PROD,
    "min": legion.LEGION_REDOP_KIND_MIN,
    "max": legion.LEGION_REDOP_KIND_MAX,
    "union": c_header.PANDAS_REDOP_RANGE_UNION,
}


@unique
class OpCode(IntEnum):
    ASTYPE = c_header.ASTYPE
    BINARY_OP = c_header.BINARY_OP
    BROADCAST_BINARY_OP = c_header.BROADCAST_BINARY_OP
    BROADCAST_FILLNA = c_header.BROADCAST_FILLNA
    BUILD_HISTOGRAM = c_header.BUILD_HISTOGRAM
    CLEAR_BITMASK = c_header.CLEAR_BITMASK
    COMPACT = c_header.COMPACT
    COMPUTE_RANGE_START = c_header.COMPUTE_RANGE_START
    COMPUTE_RANGE_STOP = c_header.COMPUTE_RANGE_STOP
    COMPUTE_RANGE_VOLUME = c_header.COMPUTE_RANGE_VOLUME
    COMPUTE_SUBRANGE_SIZES = c_header.COMPUTE_SUBRANGE_SIZES
    CONCATENATE = c_header.CONCATENATE
    CONTAINS = c_header.CONTAINS
    COPY_IF_ELSE = c_header.COPY_IF_ELSE
    COUNT_NULLS = c_header.COUNT_NULLS
    CREATE_DIR = c_header.CREATE_DIR
    DENSIFY = c_header.DENSIFY
    DROPNA = c_header.DROPNA
    DROP_DUPLICATES_CATEGORIES = c_header.DROP_DUPLICATES_CATEGORIES
    DROP_DUPLICATES_NCCL = c_header.DROP_DUPLICATES_NCCL
    DROP_DUPLICATES_TREE = c_header.DROP_DUPLICATES_TREE
    ENCODE = c_header.ENCODE
    ENCODE_CATEGORY = c_header.ENCODE_CATEGORY
    ENCODE_NCCL = c_header.ENCODE_NCCL
    EQUALS = c_header.EQUALS
    EVAL_UDF = c_header.EVAL_UDF
    EXTRACT_FIELD = c_header.EXTRACT_FIELD
    FILL = c_header.FILL
    FILLNA = c_header.FILLNA
    FINALIZE_NCCL = c_header.FINALIZE_NCCL
    FIND_BOUNDS = c_header.FIND_BOUNDS
    FIND_BOUNDS_IN_RANGE = c_header.FIND_BOUNDS_IN_RANGE
    GLOBAL_PARTITION = c_header.GLOBAL_PARTITION
    GROUPBY_REDUCE = c_header.GROUPBY_REDUCE
    IMPORT_OFFSETS = c_header.IMPORT_OFFSETS
    INIT_BITMASK = c_header.INIT_BITMASK
    INIT_NCCL = c_header.INIT_NCCL
    INIT_NCCL_ID = c_header.INIT_NCCL_ID
    ISNA = c_header.ISNA
    LIBCUDF_INIT = c_header.LIBCUDF_INIT
    LIFT_TO_DOMAIN = c_header.LIFT_TO_DOMAIN
    LOAD_PTX = c_header.LOAD_PTX
    LOCAL_HIST = c_header.LOCAL_HIST
    LOCAL_PARTITION = c_header.LOCAL_PARTITION
    MATERIALIZE = c_header.MATERIALIZE
    MERGE = c_header.MERGE
    NOTNA = c_header.NOTNA
    OFFSETS_TO_RANGES = c_header.OFFSETS_TO_RANGES
    PAD = c_header.PAD
    RANGES_TO_OFFSETS = c_header.RANGES_TO_OFFSETS
    READ_AT = c_header.READ_AT
    READ_CSV = c_header.READ_CSV
    READ_PARQUET = c_header.READ_PARQUET
    SAMPLE_KEYS = c_header.SAMPLE_KEYS
    SCALAR_BINARY_OP = c_header.SCALAR_BINARY_OP
    SCALAR_REDUCTION = c_header.SCALAR_REDUCTION
    SCAN = c_header.SCAN
    SCATTER_BY_MASK = c_header.SCATTER_BY_MASK
    SCATTER_BY_SLICE = c_header.SCATTER_BY_SLICE
    SIZES_EQUAL = c_header.SIZES_EQUAL
    SLICE_BY_RANGE = c_header.SLICE_BY_RANGE
    SORT_VALUES = c_header.SORT_VALUES
    SORT_VALUES_NCCL = c_header.SORT_VALUES_NCCL
    STRING_UOP = c_header.STRING_UOP
    STRIP = c_header.STRIP
    TO_BITMASK = c_header.TO_BITMASK
    TO_BOOLMASK = c_header.TO_BOOLMASK
    TO_BOUNDS = c_header.TO_BOUNDS
    TO_COLUMN = c_header.TO_COLUMN
    TO_CSV = c_header.TO_CSV
    TO_DATETIME = c_header.TO_DATETIME
    TO_PARQUET = c_header.TO_PARQUET
    UNARY_OP = c_header.UNARY_OP
    UNARY_REDUCTION = c_header.UNARY_REDUCTION
    WRITE_AT = c_header.WRITE_AT
    ZFILL = c_header.ZFILL


@unique
class UnaryOpCode(IntEnum):
    ABS = 0
    BIT_INVERT = 1


@unique
class BinaryOpCode(IntEnum):
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3
    FLOOR_DIV = 4
    MOD = 5
    POW = 6
    EQUAL = 7
    NOT_EQUAL = 8
    LESS = 9
    GREATER = 10
    LESS_EQUAL = 11
    GREATER_EQUAL = 12
    BITWISE_AND = 13
    BITWISE_OR = 14
    BITWISE_XOR = 15


@unique
class ProjectionCode(IntEnum):
    PROJ_RADIX_4_0 = 0
    PROJ_RADIX_4_1 = 1
    PROJ_RADIX_4_2 = 2
    PROJ_RADIX_4_3 = 3
    LAST_PROJ = 4


@unique
class KeepMethod(IntEnum):
    FIRST = 0
    LAST = 1
    NONE = 2


@unique
class JoinVariantCode(IntEnum):
    BROADCAST = 0  # broadcast-hash join
    HASH = 1  # partitioned hash join
    SORT = 2  # sort-merge join


@unique
class JoinTypeCode(IntEnum):
    INNER = 0
    LEFT = 1
    OUTER = 3


@unique
class GroupbyVariantCode(IntEnum):
    TREE = 0  # reduction tree-based groupby
    HASH = 1  # partitioned hash groupby


@unique
class AggregationCode(IntEnum):
    SUM = 0
    MIN = 1
    MAX = 2
    COUNT = 3
    PROD = 4
    MEAN = 5
    VAR = 6
    STD = 7
    SIZE = 8
    ANY = 9
    ALL = 10
    SQSUM = 11


@unique
class DatetimeFieldCode(IntEnum):
    YEAR = 0
    MONTH = 1
    DAY = 2
    HOUR = 3
    MINUTE = 4
    SECOND = 5
    WEEKDAY = 6


@unique
class PadSideCode(IntEnum):
    LEFT = 0
    RIGHT = 1
    BOTH = 2


@unique
class StringMethods(IntEnum):
    LOWER = 0
    UPPER = 1
    SWAPCASE = 2


@unique
class PartitionId(IntEnum):
    EQUAL = 100000
    WEIGHTED = 200000
    IMAGE = 300000
    # Special partitions for histogram construction
    ROW = 400000
    COLUMN = 500000


@unique
class PandasTunable(IntEnum):
    NUM_PIECES = 1
    HAS_GPUS = 2


@unique
class PandasMappingTag(IntEnum):
    BITMASK = 100
    HISTOGRAM = 101


@unique
class CompressionType(IntEnum):
    UNCOMPRESSED = 0
    SNAPPY = 1
    GZIP = 2
    BROTLI = 3
    BZ2 = 4
    ZIP = 5
    XZ = 6

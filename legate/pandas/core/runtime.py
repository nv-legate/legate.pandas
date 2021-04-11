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

import functools
import os
import struct
import sys
from collections import OrderedDict

import numpy
import pandas
from pandas.core.dtypes import common as pandas_dtype

from legate.core import (
    ArgumentMap,
    EqualPartition,
    Fence,
    FieldSpace,
    Future,
    FutureMap,
    IndexPartition,
    IndexSpace,
    IndexTask,
    LegateLibrary,
    OutputRegion,
    Partition,
    PartitionByImage,
    PartitionByImageRange,
    PartitionByRestriction,
    PartitionByWeights,
    Rect,
    Region,
    Task,
    Transform,
    ffi,
    get_legion_context,
    get_legion_runtime,
    legion,
)

from legate.pandas.common import errors as err, types as ty
from legate.pandas.common.util import to_list_if_scalar
from legate.pandas.config import (
    PANDAS_FIELD_ID_BASE,
    AggregationCode,
    BinaryOpCode,
    CompressionType,
    DatetimeFieldCode,
    JoinTypeCode,
    OpCode,
    PadSideCode,
    PandasTunable,
    PartitionId,
    ProjectionCode,
    UnaryOpCode,
    pandas_reduction_op_offsets,
)
from legate.pandas.library import library

from .bitmask import Bitmask
from .column import Column
from .future import PandasFuture, PandasFutureMap, Scalar
from .index import create_index_from_pandas, create_range_index
from .pattern import Broadcast, Map, ScalarMap
from .storage import OutputStorage, Storage
from .table import Table


# Helper method for python 3 support
def _iterkeys(obj):
    return obj.keys() if sys.version_info > (3,) else obj.viewkeys()


def _iteritems(obj):
    return obj.items() if sys.version_info > (3,) else obj.viewitems()


def _itervalues(obj):
    return obj.values() if sys.version_info > (3,) else obj.viewvalues()


# This dummy function is to load the library forcibly from clients
def load_library():
    pass


try:
    reduce  # Python 2
except NameError:
    reduce = functools.reduce

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

try:
    long  # Python 2
except NameError:
    long = int  # Python 3

UNARY_OP_NAME_TO_CODE = {
    "abs": UnaryOpCode.ABS,
    "bit_invert": UnaryOpCode.BIT_INVERT,
}

BINARY_OP_NAME_TO_CODE = {
    "add": BinaryOpCode.ADD,
    "sub": BinaryOpCode.SUB,
    "mul": BinaryOpCode.MUL,
    "div": BinaryOpCode.DIV,
    "truediv": BinaryOpCode.DIV,
    "floordiv": BinaryOpCode.FLOOR_DIV,
    "mod": BinaryOpCode.MOD,
    "pow": BinaryOpCode.POW,
    "eq": BinaryOpCode.EQUAL,
    "ne": BinaryOpCode.NOT_EQUAL,
    "lt": BinaryOpCode.LESS,
    "gt": BinaryOpCode.GREATER,
    "le": BinaryOpCode.LESS_EQUAL,
    "ge": BinaryOpCode.GREATER_EQUAL,
    "__or__": BinaryOpCode.BITWISE_OR,
    "__and__": BinaryOpCode.BITWISE_AND,
    "__xor__": BinaryOpCode.BITWISE_XOR,
}

NON_NULLABLE_BINARY_OP = set(["eq", "ne", "lt", "gt", "le", "ge"])

SCAN_OP_NAME_TO_CODE = {
    "cumsum": AggregationCode.SUM,
    "cummin": AggregationCode.MIN,
    "cummax": AggregationCode.MAX,
    "cumprod": AggregationCode.PROD,
}


PAD_SIDE_TO_CODE = {
    "left": PadSideCode.LEFT,
    "right": PadSideCode.RIGHT,
    "both": PadSideCode.BOTH,
}


class PartitionManager(object):
    def __init__(self, runtime, legion_context, legion_runtime, ispace):
        self._runtime = runtime
        self._legion_context = legion_context
        self._legion_runtime = legion_runtime
        self._ispace = ispace
        self._next_equal_partition_id = PartitionId.EQUAL.value
        self._equal_partitions = OrderedDict()
        self._next_weighted_partition_id = PartitionId.WEIGHTED.value
        self._weighted_partitions = list()
        self._next_image_partition_id = PartitionId.IMAGE.value
        self._image_partitions = list()
        self._next_row_partition_id = PartitionId.ROW.value
        self._row_partitions = OrderedDict()
        self._next_column_partition_id = PartitionId.COLUMN.value
        self._column_partitions = OrderedDict()
        # This dictionary records all primary index partitions
        # which only include equal and weighted partitions
        self._all_partitions = OrderedDict()
        self._all_weights = list()

    def find_or_create_equal_partition(self, cspace):
        if cspace not in self._equal_partitions:
            partitioner = EqualPartition()
            part_id = self._next_equal_partition_id
            assert part_id < PartitionId.WEIGHTED.value
            self._next_equal_partition_id = part_id + 1
            ipart = IndexPartition(
                self._legion_context,
                self._legion_runtime,
                self._ispace,
                cspace,
                partitioner,
                kind=legion.DISJOINT_COMPLETE_KIND,
                part_id=part_id,
                keep=True,
            )
            self._equal_partitions[cspace] = ipart

            # TODO: Here we depend on the implementation artifact that equal
            #       partitions of index spaces having the same size are
            #       isomoprhic, but this is not part of the equal
            #       partitioning's contract. In the future, we should switch
            #       to structued image partitions to preserve isomorphism.
            def creator(ispace):
                return self._runtime.create_equal_partition(ispace, cspace)

            self._all_partitions[ipart] = creator
        return self._equal_partitions[cspace]

    def find_or_create_weighted_partition(self, cspace, weights):
        # TODO: For now we don't reuse weighted partitions
        #       as they are unlikely to be reusable.
        #       We will find a good way to hash weights in the future
        #       once the lack of reusing becomes a problem.
        partitioner = PartitionByWeights(weights.future_map)
        part_id = self._next_weighted_partition_id
        assert part_id < PartitionId.IMAGE.value
        self._next_weighted_partition_id = part_id + 1
        ipart = IndexPartition(
            self._legion_context,
            self._legion_runtime,
            self._ispace,
            cspace,
            partitioner,
            kind=legion.DISJOINT_COMPLETE_KIND,
            part_id=part_id,
            keep=True,
        )
        self._weighted_partitions.append(ipart)

        def creator(ispace):
            return self._runtime.create_partition_from_weights(
                ispace, cspace, weights
            )

        self._all_partitions[ipart] = creator
        return ipart

    def find_or_create_image_partition(
        self, cspace, src_part, field_id, kind, range
    ):
        ctor = PartitionByImageRange if range else PartitionByImage
        partitioner = ctor(
            src_part.parent,
            src_part,
            field_id,
            self._runtime.mapper_id,
        )
        part_id = self._next_image_partition_id
        self._next_image_partition_id = part_id + 1
        ipart = IndexPartition(
            self._legion_context,
            self._legion_runtime,
            self._ispace,
            cspace,
            partitioner,
            kind=kind,
            part_id=part_id,
        )
        self._image_partitions.append(ipart)
        return ipart

    def find_or_create_row_partition(self, cspace, num_rows):
        if cspace not in self._row_partitions:
            transform = Transform(2, 1)
            transform.trans[0, 0] = 0
            transform.trans[1, 0] = 1
            extent = Rect([num_rows, 1])
            partitioner = PartitionByRestriction(transform, extent)
            part_id = self._next_row_partition_id
            self._next_row_partition_id = part_id + 1
            ipart = IndexPartition(
                self._legion_context,
                self._legion_runtime,
                self._ispace,
                cspace,
                partitioner,
                kind=legion.DISJOINT_COMPLETE_KIND,
                part_id=part_id,
            )
            self._row_partitions[cspace] = ipart
        return self._row_partitions[cspace]

    def find_or_create_column_partition(self, cspace, num_columns):
        if cspace not in self._column_partitions:
            transform = Transform(2, 1)
            transform.trans[0, 0] = 1
            transform.trans[1, 0] = 0
            extent = Rect([1, num_columns])
            partitioner = PartitionByRestriction(transform, extent)
            part_id = self._next_column_partition_id
            self._next_column_partition_id = part_id + 1
            ipart = IndexPartition(
                self._legion_context,
                self._legion_runtime,
                self._ispace,
                cspace,
                partitioner,
                kind=legion.DISJOINT_COMPLETE_KIND,
                part_id=part_id,
            )
            self._column_partitions[cspace] = ipart
        return self._column_partitions[cspace]

    def create_isomorphic_partition(self, ipart, ispace):
        assert ipart in self._all_partitions
        return self._all_partitions[ipart](ispace)

    def register_external_partition(self, ipart, creator):
        assert ipart not in self._all_partitions
        self._all_partitions[ipart] = creator

    def is_internal_partition(self, ipart):
        return ipart in self._all_partitions


class Runtime(LegateLibrary):
    def __init__(self, context, runtime):
        self._context = context
        self._runtime = runtime

        library.set_runtime(self)

        # Generate IDs
        (
            self._first_task_id,
            self._first_functor_id,
            self._first_redop_id,
            self.mapper_id,
        ) = library.generate_ids(self._runtime)

        # Maintain Legion objects for clean-up
        self._index_spaces = (
            OrderedDict()
        )  # index spaces de-duplicated by size
        self._color_spaces = (
            OrderedDict()
        )  # color spaces de-duplicated by size
        self._field_spaces = list()  # list of field spaces
        self._partitions = OrderedDict()
        self._external_storages = OrderedDict()
        self._storages = list()

        # maps index spaces to partition managers
        self._partition_managers = OrderedDict()

        # Initialize global constants
        self.num_pieces = self._get_tunable(PandasTunable.NUM_PIECES, int)
        self.has_gpus = self._get_tunable(PandasTunable.HAS_GPUS, bool)
        if self.has_gpus:
            self.cuda_arch = library.get_cuda_arch()
            self.use_nccl = library.get_use_nccl()
        else:
            self.use_nccl = False
        self.radix = 4
        self.empty_argmap = legion.legion_argument_map_create()
        self.debug = self._is_variable_set("DEBUG_LEGATE")
        self.trace_storages = self._is_variable_set("TRACE_STORAGES")

        if self.use_nccl:
            self._initialize_nccl()
        if self.has_gpus:
            self._preload_libcudf()
        self._cache_legion_types()

    @property
    def pandas_field_id_base(self):
        return PANDAS_FIELD_ID_BASE

    @staticmethod
    def _is_variable_set(variable):
        return variable in os.environ and int(os.environ[variable]) > 0

    def _get_tunable(self, tunable_id, cast):
        f = Future(
            legion.legion_runtime_select_tunable_value(
                self._runtime,
                self._context,
                tunable_id,
                self.mapper_id,
                0,
            )
        )
        return cast(struct.unpack_from("i", f.get_buffer(4))[0])

    def _initialize_nccl(self):
        task = Task(
            self.get_task_id(OpCode.INIT_NCCL_ID),
            mapper=self.mapper_id,
        )
        self._nccl_id = self.dispatch(task)

        task = IndexTask(
            self.get_task_id(OpCode.INIT_NCCL),
            Rect([self.num_pieces]),
            argmap=self.empty_argmap,
            mapper=self.mapper_id,
        )
        task.add_future(self._nccl_id)
        self.issue_fence()
        self._nccl_comm = self.dispatch(task).cast(ty.uint64)
        self.issue_fence()

    def _finalize_nccl(self):
        task = IndexTask(
            self.get_task_id(OpCode.FINALIZE_NCCL),
            Rect([self.num_pieces]),
            argmap=self.empty_argmap,
            mapper=self.mapper_id,
        )
        nccl_comm = self._nccl_comm._future_map
        task.add_point_future(ArgumentMap(future_map=nccl_comm))
        self.dispatch(task).wait()

    def _preload_libcudf(self):
        task = IndexTask(
            self.get_task_id(OpCode.LIBCUDF_INIT),
            Rect([self.num_pieces]),
            argmap=self.empty_argmap,
            mapper=self.mapper_id,
        )
        self.dispatch(task).wait()

    def _cache_legion_types(self):
        legion_types = [
            "legion_domain_t *",
            "legion_domain_point_t *",
            "legion_domain_transform_t *",
            "legion_index_space_t *",
            "legion_index_partition_t *",
            "legion_logical_region_t *",
            "legion_logical_partition_t *",
            "legion_point_1d_t *",
            "legion_rect_1d_t *",
            "legion_rect_2d_t *",
            "legion_task_argument_t *",
            "legion_transform_1x1_t *",
            "legion_transform_2x1_t *",
            "legion_transform_1x2_t *",
            "legion_transform_2x2_t *",
            "legion_output_requirement_t *",
            "legion_region_requirement_t *",
        ]
        for t in legion_types:
            ffi.new(t)

    def destroy(self):
        if self.use_nccl:
            self._finalize_nccl()
            del self._nccl_id
            del self._nccl_comm
        self._storages.clear()
        self._external_storages.clear()
        legion.legion_argument_map_destroy(self.empty_argmap)
        self._empty_argmap = None
        self._partition_managers.clear()
        self._field_spaces.clear()

    def get_unary_op_code(self, op):
        return UNARY_OP_NAME_TO_CODE[op]

    def get_binary_op_code(self, op):
        return BINARY_OP_NAME_TO_CODE[op]

    def is_nullable_binary_op(self, op):
        return op not in NON_NULLABLE_BINARY_OP

    def get_scan_op_code(self, op):
        return SCAN_OP_NAME_TO_CODE[op]

    def get_datetime_field_code(self, field):
        return getattr(DatetimeFieldCode, field.upper())

    def get_pad_side_code(self, side):
        return PAD_SIDE_TO_CODE[side]

    def get_compression_type(self, compression):
        if compression is None:
            return CompressionType.UNCOMPRESSED
        else:
            return getattr(CompressionType, compression.upper())

    def get_task_id(self, op_code):
        return self._first_task_id + op_code.value

    def get_reduction_op_id(self, op, argument_type):
        redop_id = pandas_reduction_op_offsets[op]
        if op == "union" and ty.is_range_dtype(argument_type):
            return self._first_redop_id + redop_id

        result = legion.LEGION_REDOP_BASE + redop_id * legion.LEGION_TYPE_TOTAL
        result += ty.encode_dtype(argument_type)
        return result

    def get_projection_functor_id(self, functor_code):
        return self._first_functor_id + functor_code.value

    def get_radix_functor_id(self, radix, offset):
        if radix == 1:
            return 0
        else:
            proj_name = "PROJ_RADIX_%d_%d" % (radix, offset)
            return self.get_projection_functor_id(
                getattr(ProjectionCode, proj_name)
            )

    def get_join_type_id(self, how):
        return getattr(JoinTypeCode, how.upper())

    def get_partition(self, region, ipart):
        key = (region, ipart)
        if key in self._partitions:
            return self._partitions[key]
        else:
            partition = Partition(self._context, self._runtime, ipart, region)
            self._partitions[key] = partition
            return partition

    def find_or_create_partition_manager(self, ispace):
        if ispace not in self._partition_managers:
            self._partition_managers[ispace] = PartitionManager(
                self, self._context, self._runtime, ispace
            )
        return self._partition_managers[ispace]

    def _create_column_from_pandas(self, storage, pandas_series):
        dtype = pandas_series.dtype
        if pandas_dtype.is_string_dtype(dtype):
            return self._create_string_column_from_pandas(
                storage, pandas_series
            )
        elif pandas_dtype.is_categorical_dtype(dtype):
            return self._create_category_column_from_pandas(
                storage, pandas_series
            )
        else:
            return self._create_numeric_column_from_pandas(
                storage, pandas_series
            )

    def _create_string_column_from_pandas(
        self, storage, pandas_series, num_pieces=None
    ):
        import pyarrow

        # Convert the string series to an Arrow array to access raw data
        # more easily
        if any(
            not (isinstance(v, str) or pandas.isna(v)) for v in pandas_series
        ):
            raise err._unsupported_error(
                "Series with mixed type values are not supported yet"
            )
        array = pyarrow.array(pandas_series)
        buffers = array.buffers()
        if buffers == [None]:
            # If we are here, then all entries are nulls.
            column_ipart = self.create_equal_partition(storage.ispace, 1)
            column = storage.create_column(
                ty.string,
                ipart=column_ipart,
                nullable=True,
            )
            column.bitmask._storage.fill(0)

            size = len(pandas_series)
            offsets_storage = self.create_storage(size + 1 if size > 0 else 0)
            offsets_ipart = self.create_equal_partition(
                offsets_storage.ispace, 1
            )
            offsets = offsets_storage.create_column(
                ty.int32,
                ipart=offsets_ipart,
                nullable=False,
            )
            column.add_child(offsets)
            offsets.data.fill(0)

            chars_storage = self.create_storage(0)
            chars_ipart = self.create_equal_partition(chars_storage.ispace, 1)
            chars = chars_storage.create_column(
                ty.int8,
                ipart=chars_ipart,
                nullable=False,
            )
            column.add_child(chars)
            chars.data.fill(0)

        else:
            pa_bitmask, pa_offsets, pa_chars = buffers

            column_ipart = self.create_equal_partition(storage.ispace, 1)
            column = storage.create_column(
                ty.string,
                ipart=column_ipart,
                nullable=pa_bitmask is not None,
            )

            # FIXME: For now we import string columns sequentially so that we
            #        don't have to deal with partitioning regions that are
            #        not aligned with each other. We will encourage users
            #        to rather use one of the IO functions, such as read_csv,
            #        because they will be much more scalable and efficient.

            # TODO: We may want to cache these temporary storages to make
            #       this code less bad
            size = len(pandas_series)
            offsets_storage = self.create_storage(size + 1 if size > 0 else 0)
            offsets_ipart = self.create_equal_partition(
                offsets_storage.ispace, 1
            )

            temp_offsets = offsets_storage.create_new_field(ty.int32)
            temp_offsets.from_arrow(pa_offsets)

            offsets_ipart = self.create_equal_partition(
                offsets_storage.ispace, 1
            )
            offsets = offsets_storage.create_column(
                ty.int32,
                ipart=offsets_ipart,
                nullable=False,
            )
            column.add_child(offsets)

            # Import the bitmask only when it exists
            temp_bitmask = None
            if column.nullable:
                temp_bitmask = self.create_storage(
                    pa_bitmask.size
                ).create_new_field(Bitmask.alloc_type)
                temp_bitmask.from_arrow(pa_bitmask)

            # Convert Arrow bitmask and offsets to Legate bitmask and ranges
            plan = Map(self, OpCode.IMPORT_OFFSETS)

            column.add_to_plan(plan, False, proj=None)
            plan.add_input(temp_offsets, Broadcast)
            if column.nullable:
                plan.add_input(temp_bitmask, Broadcast)
            plan.execute_single()

            chars_storage = self.create_storage(pa_chars.size)
            chars_ipart = self.create_equal_partition(chars_storage.ispace, 1)
            chars = chars_storage.create_column(
                ty.int8,
                ipart=chars_ipart,
                nullable=False,
            )
            chars.data.from_arrow(pa_chars)
            column.add_child(chars)

        if num_pieces is None:
            num_pieces = self.num_pieces
        return column.repartition(num_pieces).as_string_column()

    def _create_category_column_from_pandas(self, storage, pandas_series):
        dtype = ty.CategoricalDtype.from_pandas(self, pandas_series.dtype)
        result_column = storage.create_column(
            dtype,
            nullable=pandas_series.hasnans,
        )

        codes = pandas_series.cat.codes.astype(numpy.uint32)
        result_column.add_child(
            self._create_numeric_column_from_pandas(storage, codes)
        )
        result_column.add_child(dtype.categories_column)

        result_column = result_column.as_category_column()
        if result_column.nullable:
            result_column.initialize_bitmask()

        return result_column

    def _create_numeric_column_from_pandas(self, storage, pandas_series):
        dtype = ty.to_legate_dtype(pandas_series.dtype)
        column = storage.create_column(dtype, nullable=pandas_series.hasnans)
        pandas_values = pandas_series.values
        if not (
            pandas_values.flags["C_CONTIGUOUS"]
            or pandas_values.flags["F_CONTIGUOUS"]
        ):
            pandas_values = pandas.Series(pandas_series, copy=True).values
        column.data.from_numpy(pandas_values)
        if column.nullable:
            column.initialize_bitmask()
        return column

    def create_dataframe_from_legate_data(self, legate_data):
        columns = []
        for field, array in legate_data.items():
            stores = array.stores()
            columns.append(Column.from_stores(field.type, stores))

        if any(columns[0].ispace is not column.ispace for column in columns):
            raise err._unsupported_error(
                "All Legate Arrays must have the same index space for now"
            )

        # See if there is any partition that we can reuse
        primary_ipart = None
        if columns[0].ispace.children is not None:
            for ipart in columns[0].ispace.children:
                if self.is_internal_partition(ipart):
                    continue
                functor = ipart.functor
                if functor is None:
                    continue
                if isinstance(functor, EqualPartition):
                    primary_ipart = ipart
                    self.register_external_equal_partition(ipart)
                    break
                elif isinstance(functor, PartitionByWeights):
                    primary_ipart = ipart
                    self.register_external_weighted_partition(
                        ipart, functor.weights
                    )
                    break

        # If we haven't found a partition yet, we use the default partition
        # of the first column' storage
        if primary_ipart is None:
            primary_ipart = columns[0].storage.default_ipart

        for column in columns:
            column.set_primary_ipart(primary_ipart)

        index_volume = self._get_index_space_volume(columns[0].ispace)
        index_volume = self.create_future(index_volume, ty.int64)
        index = create_range_index(columns[0].storage, index_volume)

        return Table(self, index, columns)

    def create_dataframe_from_pandas(self, pandas_obj, index=None):
        assert isinstance(pandas_obj, (pandas.DataFrame, pandas.Series))

        if index is not None:
            storage = index.storage
        else:
            storage = self.create_storage(len(pandas_obj.index))
            index = create_index_from_pandas(self, storage, pandas_obj.index)

        if isinstance(pandas_obj, pandas.DataFrame):
            columns = [
                self._create_column_from_pandas(storage, pandas_series)
                for (_, pandas_series) in pandas_obj.items()
            ]
        else:
            columns = [self._create_column_from_pandas(storage, pandas_obj)]

        return Table(self, index, columns)

    def create_storage(self, ispace, ipart=None):
        if not isinstance(ispace, IndexSpace):
            ispace = self.find_or_create_index_space(ispace)
        storage = Storage(self, ispace, ipart=ipart)
        if self.trace_storages:
            self._storages.append(storage)
        return storage

    def create_output_storage(self):
        return OutputStorage(self)

    def create_output_region(
        self, storage, fields, global_indexing=True, ipart=None
    ):
        if storage.fixed:
            assert ipart is not None
            partition = self.get_partition(storage.region, ipart)
            # TODO: We will always use the identity projection for now
            region = OutputRegion(
                self._context,
                self._runtime,
                fields=fields,
                existing=partition,
                proj=0,
                flags=legion.LEGION_CREATED_OUTPUT_REQUIREMENT_FLAG,
            )
        else:
            region = OutputRegion(
                self._context,
                self._runtime,
                field_space=storage.fspace,
                fields=fields,
                global_indexing=global_indexing,
            )
        return region

    def dump_storage_stat(self):
        storages = [storage for storage in self._storages if not storage.empty]
        if len(storages) == 0:
            return
        total_bytes = 0
        total_free_bytes = 0
        print("%d storages:" % len(storages))
        for storage in storages:
            bytes, free_bytes = storage.dump_stat()
            total_bytes += bytes
            total_free_bytes += free_bytes
        print(
            "%.2f MBs in total, %.2f MBs unused (%.1f %% used)"
            % (
                total_bytes / 1000000.0,
                total_free_bytes / 1000000.0,
                (total_bytes - total_free_bytes) / total_bytes * 100,
            )
        )

    def _create_external_storage(self, region):
        if region in self._external_storages:
            return self._external_storages[region]
        else:
            external_storage = Storage(self, region=region, external=True)
            self._external_storages[region] = external_storage
            return external_storage

    def _get_index_space_volume(self, ispace):
        bounds = ispace.get_bounds()
        if self.debug:
            assert bounds.dim == 1
        return (bounds.hi[0] - bounds.lo[0] + 1,)

    def _create_index_space(self, rect):
        if not isinstance(rect, PandasFuture):
            if not isinstance(rect, Rect):
                rect = Rect([rect])
            handle = legion.legion_index_space_create_domain(
                self._runtime, self._context, rect.raw()
            )
        else:
            domain = self.launch_future_task(OpCode.LIFT_TO_DOMAIN, rect)
            handle = legion.legion_index_space_create_future(
                self._runtime, self._context, 1, domain.handle, 0
            )
        return IndexSpace(self._context, self._runtime, handle=handle)

    def find_or_create_index_space(self, num_elements):
        if isinstance(num_elements, PandasFuture) and num_elements.ready:
            num_elements = long(num_elements.get_value())
        if (
            not isinstance(num_elements, PandasFuture)
            and num_elements in self._index_spaces
        ):
            return self._index_spaces[num_elements]
        ispace = self._create_index_space(num_elements)
        if not isinstance(num_elements, PandasFuture):
            self._index_spaces[num_elements] = ispace
        return ispace

    def find_or_create_color_space(self, num_colors):
        if num_colors in self._color_spaces:
            return self._color_spaces[num_colors]

        colors = self._create_index_space(num_colors)
        self._color_spaces[num_colors] = colors
        return colors

    def create_field_space(self):
        fspace = FieldSpace(self._context, self._runtime)
        self._field_spaces.append(fspace)
        return fspace

    def create_logical_region(self, ispace, fspace):
        handle = legion.legion_logical_region_create(
            self._runtime,
            self._context,
            ispace.handle,
            fspace.handle,
            True,
        )
        region = Region(self._context, self._runtime, ispace, fspace, handle)
        return region

    def create_future(self, value, dtype=None):
        if ty.is_categorical_dtype(dtype):
            dtype = ty.string
        if ty.is_string_dtype(dtype):
            return self.create_future_from_string(value)
        pandas_dtype = (
            dtype.storage_dtype.to_pandas() if dtype is not None else None
        )
        result = Future()
        value = numpy.array(value, dtype=pandas_dtype)
        if ty.is_timestamp_dtype(dtype):
            value = value.view(dtype.storage_dtype.to_pandas())
        result.set_value(self._runtime, value.data, value.nbytes)
        return PandasFuture(self, result, dtype, ready=True)

    def create_future_from_scalar(self, scalar):
        if scalar.valid:
            if ty.is_string_dtype(scalar.dtype):
                size = len(scalar._value)
                buf = struct.pack(
                    f"iiQ{size}s",
                    scalar.valid,
                    ty.encode_dtype(scalar.dtype),
                    size,
                    scalar._value.encode("utf-8"),
                )
            else:
                fmt = ty.to_format_string(scalar.dtype.storage_dtype)
                buf = struct.pack(
                    "ii" + fmt,
                    scalar.valid,
                    ty.encode_dtype(scalar.dtype),
                    scalar._value,
                )
        else:
            buf = struct.pack(
                "iiQ", scalar.valid, ty.encode_dtype(scalar.dtype), 0
            )

        fut = Future()
        fut.set_value(self._runtime, buf, len(buf))
        return PandasFuture(self, fut, scalar.dtype, True)

    def create_scalar(self, value, dtype):
        if ty.is_categorical_dtype(dtype):
            dtype = ty.string

        # Sanitize the value to make it fit to the dtype
        if value is not None:
            if isinstance(value, numpy.datetime64):
                value = value.view("int64")

            if ty.is_string_dtype(dtype):
                value = str(value)
            elif ty.is_integer_dtype(dtype):
                value = int(value)

        return Scalar(
            self, dtype, value is not None, 0 if value is None else value
        )

    def create_future_from_string(self, value):
        bytes = value.encode("utf-8")
        result = Future()
        result.set_value(self._runtime, bytes, len(bytes))
        return PandasFuture(self, result, dtype=ty.string, ready=True)

    def create_partition_from_weights(self, ispace, colors, weights):
        return self.find_or_create_partition_manager(
            ispace
        ).find_or_create_weighted_partition(colors, weights)

    def create_equal_partition(self, ispace, colors):
        if type(colors) != IndexSpace:
            colors = self.find_or_create_color_space(colors)
        return self.find_or_create_partition_manager(
            ispace
        ).find_or_create_equal_partition(colors)

    def create_partition_by_image(
        self,
        ispace,
        colors,
        storage,
        storage_ipart,
        kind=legion.COMPUTE_KIND,
        range=False,
    ):
        partition = storage.get_view(storage_ipart)
        return self.find_or_create_partition_manager(
            ispace
        ).find_or_create_image_partition(
            colors, partition, storage.field_id, kind, range
        )

    def create_row_partition(self, ispace, colors, num_rows):
        return self.find_or_create_partition_manager(
            ispace
        ).find_or_create_row_partition(colors, num_rows)

    def create_column_partition(self, ispace, colors, num_rows):
        return self.find_or_create_partition_manager(
            ispace
        ).find_or_create_column_partition(colors, num_rows)

    def create_isomorphic_partition(self, ispace, to_copy):
        return self.find_or_create_partition_manager(
            to_copy.parent
        ).create_isomorphic_partition(to_copy, ispace)

    def register_external_partition(self, ipart, creator):
        self.find_or_create_partition_manager(
            ipart.parent
        ).register_external_partition(ipart, creator)

    def register_external_equal_partition(self, ipart):
        def creator(ispace):
            return self.create_equal_partition(ispace, ipart.color_space)

        self.register_external_partition(ipart, creator)

    def register_external_weighted_partition(self, ipart, weights):
        def creator(ispace):
            return self.create_partition_from_weights(
                ispace, ipart.color_space, weights
            )

        self.register_external_partition(ipart, creator)

    def is_internal_partition(self, ipart):
        return self.find_or_create_partition_manager(
            ipart.parent
        ).is_internal_partition(ipart)

    def create_region_partition(self, region, ipart):
        return Partition(self._context, self._runtime, ipart, region)

    def unmap_physical_region(self, pr):
        pr.unmap(self._runtime, self._context, unordered=True)

    def launch_future_task(self, op_code, futures, dtype=None):
        task = Task(self.get_task_id(op_code))
        futures = to_list_if_scalar(futures)
        for future in futures:
            task.add_future(future)
        result = self.dispatch(task)
        if dtype is not None:
            result = result.cast(dtype)
        return result

    def launch_future_map_task(self, op_code, future_map, dtype=None):
        launch_domain = legion.legion_future_map_get_domain(future_map.handle)
        task = IndexTask(
            self.get_future_task_id(op_code),
            launch_domain,
            argmap=legion.legion_argument_map_from_future_map(
                future_map.handle
            ),
        )
        result = self.dispatch(task)
        if dtype is not None:
            result = result.cast(dtype)
        return result

    def all(self, futures):
        not_ready = []
        for future in futures:
            if not future.ready:
                not_ready.append(future)
            elif not future.get_scalar().value:
                return self.create_scalar(False, ty.bool).get_future()

        if len(not_ready) == 0:
            return self.create_scalar(True, ty.bool).get_future()
        elif len(not_ready) == 1:
            return not_ready[0]

        plan = ScalarMap(self, OpCode.SCALAR_REDUCTION, ty.bool)
        plan.add_scalar_arg(ty.get_aggregation_op_id("all"), ty.int32)
        for future in not_ready:
            plan.add_future(future)
        return plan.execute_single()

    def reduce_future_map(self, future_map, op, dtype, deterministic):
        redop = self.get_reduction_op_id(op, dtype)
        return future_map.reduce(
            self._context,
            self._runtime,
            redop,
            deterministic,
            mapper=self.mapper_id,
        )

    def issue_fence(self, block=False):
        fence = Fence(False)
        f = self.dispatch(fence)
        if block:
            f.wait()

    def _get_shard_id(self):
        # TODO: We don't need to use the shard id once we move everything
        #       we want to do only on node 0 to Python tasks
        return int(
            legion.legion_context_get_shard_id(
                self._runtime, self._context, True
            )
        )

    def _this_is_first_node(self):
        return self._get_shard_id() == 0

    def dispatch(self, operation, redop=None, unordered=None):
        if unordered is None:
            if redop is None:
                value = operation.launch(self._runtime, self._context)
            else:
                value = operation.launch(self._runtime, self._context, redop)

            if value is not None:
                if isinstance(value, Future):
                    return PandasFuture(self, value)
                elif isinstance(value, FutureMap):
                    return PandasFutureMap(self, value)
                else:
                    return value
        else:
            assert redop is None
            operation.launch(self._runtime, self._context, unordered)


_runtime = Runtime(get_legion_context(), get_legion_runtime())

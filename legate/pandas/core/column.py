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

import warnings

import numpy as np
import pandas as pd
import pyarrow as pa

from legate.core import FieldID, Rect, Region, legion

from legate.pandas.common import errors as err, types as ty
from legate.pandas.config import OpCode, StringMethods

from .future import PandasFuture, Scalar
from .pattern import Broadcast, Map, Projection, ScalarMap


def _create_column(storage, dtype, ipart=None, nullable=False):
    column = storage.create_column(dtype, ipart=ipart, nullable=nullable)
    if ty.is_string_dtype(dtype):
        offset_storage = storage._runtime.create_output_storage()
        char_storage = storage._runtime.create_output_storage()
        column.add_child(
            offset_storage.create_column(ty.int32, nullable=False)
        )
        column.add_child(char_storage.create_column(ty.int8, nullable=False))
        column = column.as_string_column()
    return column


class Column(object):
    def __init__(self, runtime, data=None, bitmask=None, children=[]):
        self.runtime = runtime
        self.data = data
        self.bitmask = bitmask
        self.children = children.copy()

    ########################################
    # LegateArray methods
    ##############################

    def stores(self):
        stores = []

        if self.nullable:
            stores.append(self.bitmask._storage.legate_store)
        else:
            stores.append(None)

        if self.data is not None:
            stores.append(self.data.legate_store)

        for child in self.children:
            stores.extend(child.stores())

        return stores

    @staticmethod
    def _import_store(rt, store):
        if store is None:
            return None
        kind = store.kind

        if kind not in ((Region, FieldID), (Region, int)):
            raise err._unsupported_error(
                f"Unsupported Legate Store kind: {kind}"
            )

        (region, fid) = store.storage

        if region.index_space.get_dim() != 1:
            raise err._unsupported_error("All Legate Arrays must be 1-D")

        dtype = ty.to_legate_dtype(store.type)
        if kind[1] is FieldID:
            fid = fid.fid

        storage = rt._create_external_storage(region)
        return storage.import_field(region, fid, dtype)

    @staticmethod
    def from_stores(type, stores, children=None):
        from .bitmask import Bitmask
        from .runtime import _runtime as rt

        if children is not None:
            raise err._unsupported_error("Only accept flat stores for now")

        slices = [Column._import_store(rt, store) for store in stores]

        if len(stores) > 2:
            raise err._unsupported_error(
                f"Unsupported Legate Array type: {type}"
            )

        dtype = ty.to_legate_dtype(type)

        assert dtype == slices[1].dtype

        bitmask = None if slices[0] is None else Bitmask(rt, slices[0])
        return Column(rt, slices[1], bitmask)

    def type(self):
        return self.dtype.to_arrow()

    def region(self):
        if self.data is None:
            return None
        else:
            return self.data.region

    ########################################

    @property
    def partitioned(self):
        return True

    @property
    def nullable(self):
        return self.bitmask is not None

    def add_child(self, column):
        self.children.append(column)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def storage(self):
        return self.data.storage

    @property
    def ispace(self):
        return self.storage.ispace

    @property
    def primary_ipart(self):
        return self.data.primary_ipart

    @property
    def num_pieces(self):
        return self.data.num_pieces

    @property
    def launch_domain(self):
        return self.data.launch_domain

    @property
    def cspace(self):
        return self.primary_ipart.color_space

    def clone(self, shallow=False):
        data = None if self.data is None else self.data.clone()
        bitmask = None if self.bitmask is None else self.bitmask.clone()
        children = (
            self.children
            if shallow
            else [child.clone() for child in self.children]
        )
        return type(self)(self.runtime, data, bitmask, children)

    def set_storage(self, storage):
        self._storage = storage

    def has_nulls(self):
        if self.bitmask is None:
            return False
        else:
            return self.bitmask.has_nulls()

    def is_isomorphic(self, other):
        assert isinstance(self, type(other))
        return self.primary_ipart == other.primary_ipart

    def set_primary_ipart(self, ipart, recurse=True):
        self.data.set_primary_ipart(ipart)

        if self.nullable:
            self.bitmask.set_primary_ipart(ipart)

        if recurse:
            if ty.is_string_dtype(self.dtype):
                ranges = self.children[0]
                values = self.children[1]

                ranges.set_primary_ipart(ipart)

                assert ty.is_range_dtype(ranges.data.dtype)
                values.set_primary_ipart(
                    self.runtime.create_partition_by_image(
                        values.ispace,
                        self.cspace,
                        ranges.data,
                        ipart,
                        kind=legion.DISJOINT_COMPLETE_KIND,
                        range=True,
                    )
                )
            else:
                for child in self.children:
                    child.set_primary_ipart(ipart)

    def repartition_by_ipart(self, new_ipart):
        new_self = self.all_to_ranges().clone()

        new_self.set_primary_ipart(new_ipart)

        new_self = new_self.all_to_offsets()

        return new_self

    def set_bitmask(self, bitmask):
        assert self.bitmask is None
        self.bitmask = bitmask

    def null_count(self):
        return self.bitmask.null_count()

    def set_non_nullable(self):
        self.bitmask = None

    def initialize_bitmask(self):
        assert ty.is_nullable_dtype(self.dtype)

        rt = self.runtime

        plan_init = Map(rt, OpCode.INIT_BITMASK)

        null_value = rt.create_scalar(ty.null_value(self.dtype), self.dtype)
        null_value.add_to_plan(plan_init)

        bitmask = Column(rt, self.bitmask._storage)
        bitmask.add_to_plan_output_only(plan_init)

        new_self = type(self)(rt, self.data)
        new_self.add_to_plan(plan_init, True)

        plan_init.execute(self.launch_domain)

    @staticmethod
    def _get_projection(storage, proj):
        if proj is None:
            return Broadcast
        else:
            return Projection(storage.primary_ipart, proj)

    def add_to_plan(self, plan, read, proj=0):
        if read:
            f = plan.add_input
        elif read is None:
            f = plan.add_inout
        else:
            f = plan.add_output

        f(self.data, self._get_projection(self.data, proj))
        plan.add_scalar_arg(self.nullable, ty.bool)
        if self.nullable:
            f(self.bitmask, self._get_projection(self.bitmask, proj))
        plan.add_scalar_arg(len(self.children), ty.uint32)
        for child in self.children:
            child.add_to_plan(plan, read, proj)

    def add_to_plan_output_only(self, plan):
        plan.add_output_only(self.data)
        plan.add_scalar_arg(self.nullable, ty.bool)
        if self.nullable:
            plan.add_output_only(self.bitmask)
        plan.add_scalar_arg(len(self.children), ty.uint32)
        for child in self.children:
            child.add_to_plan_output_only(plan)

    def fill(self, value, volume):
        value = self.runtime.create_scalar(value, self.dtype)

        plan = Map(self.runtime, OpCode.FILL)

        self.add_to_plan_output_only(plan)
        plan.add_future(volume)
        value.add_to_plan(plan)
        plan.add_scalar_arg(self.num_pieces, ty.int32)
        plan.execute(self.launch_domain)

    def read_at(self, idx):
        result_storage = self.runtime.create_output_storage()
        result_column = result_storage.create_similar_column(self)

        plan = Map(self.runtime, OpCode.READ_AT)

        plan.add_future(idx)
        result_column.add_to_plan_output_only(plan)
        self.add_to_plan(plan, True)

        counts = plan.execute(self.launch_domain)
        result_storage = plan.promote_output_storage(result_storage)
        self.runtime.register_external_weighted_partition(
            result_storage.default_ipart, counts
        )

        return result_column

    def write_at(self, idx, val):
        # FIXME: We copy the whole column for now, but we should perform
        #        an in-place update for primitive types.
        nullable = self.nullable or val is None
        result_column = self.storage.create_similar_column(
            self, nullable=nullable
        )
        val = self.runtime.create_scalar(val, self.dtype)

        plan = Map(self.runtime, OpCode.WRITE_AT)

        plan.add_future(idx)
        result_column.add_to_plan_output_only(plan)
        self.add_to_plan(plan, True)
        val.add_to_plan(plan)
        plan.execute(self.launch_domain)

        return result_column

    def astype(self, result_dtype, **kwargs):
        if result_dtype == self.dtype:
            return self
        elif ty.is_timestamp_dtype(result_dtype):
            if self.dtype != ty.string:
                raise err._unsupported_error(
                    f"astype to {result_dtype} is not yet supported. "
                    "please use to_datetime instead"
                )
            else:
                format = "%Y-%m-%d %H:%M:%S"
                warnings.warn(
                    f"astype from {self.dtype} to {result_dtype} currently "
                    f"uses a fixed format string '{format}' to parse strings. "
                    "please use to_datetime instead if you want the strings "
                    "to be parsed differently."
                )
                return self.to_datetime(format)

        if self.dtype == ty.ts_ns:
            if not (result_dtype == ty for ty in (ty.int64, ty.string)):
                raise TypeError(
                    "cannot astype a datetimelike from "
                    f"datetime64[ns] to {result_dtype}"
                )

            if result_dtype == ty.int64:
                return self.cast_unsafe(result_dtype)

        runtime = self.runtime

        result = self.storage.create_column(
            result_dtype, ipart=self.primary_ipart, nullable=False
        )
        if ty.is_string_dtype(result_dtype):
            offsets_storage = runtime.create_output_storage()
            chars_storage = runtime.create_output_storage()
            result.add_child(
                offsets_storage.create_column(ty.int32, nullable=False)
            )
            result.add_child(
                chars_storage.create_column(ty.int8, nullable=False)
            )
            result = result.as_string_column()

        plan = Map(runtime, OpCode.ASTYPE)

        result.add_to_plan_output_only(plan)
        self.add_to_plan(plan, True)

        plan.execute(result.launch_domain)

        result.set_bitmask(self.bitmask)

        return result

    def get_dt_field(self, field, result_dtype):
        rt = self.runtime

        result = self.storage.create_column(
            result_dtype, ipart=self.primary_ipart, nullable=False
        )

        plan = Map(rt, OpCode.EXTRACT_FIELD)

        plan.add_scalar_arg(rt.get_datetime_field_code(field), ty.int32)
        result.add_to_plan_output_only(plan)
        self.add_to_plan(plan, True)

        plan.execute(result.launch_domain)

        result.set_bitmask(self.bitmask)

        return result

    def binary_op(self, op, rhs, lhs_dtype, swapped=False, **kwargs):
        runtime = self.runtime

        if swapped:
            rhs1, rhs2 = rhs, self
        else:
            rhs1, rhs2 = self, rhs

        rhs1_scalar = isinstance(rhs1, Scalar) or isinstance(
            rhs1, PandasFuture
        )
        rhs2_scalar = isinstance(rhs2, Scalar) or isinstance(
            rhs2, PandasFuture
        )

        if rhs1_scalar or rhs2_scalar:
            if rhs1_scalar:
                nullable = rhs2.nullable and runtime.is_nullable_binary_op(op)
                lhs = rhs2.storage.create_column(
                    lhs_dtype, ipart=rhs2.primary_ipart, nullable=nullable
                )
            else:
                nullable = rhs1.nullable and runtime.is_nullable_binary_op(op)
                lhs = rhs1.storage.create_column(
                    lhs_dtype, ipart=rhs1.primary_ipart, nullable=nullable
                )
        else:
            lhs_nullable = (
                rhs1.nullable or rhs2.nullable
            ) and runtime.is_nullable_binary_op(op)
            lhs = rhs1.storage.create_column(
                lhs_dtype, ipart=rhs1.primary_ipart, nullable=lhs_nullable
            )

        assert rhs1.dtype == rhs2.dtype or (
            ty.is_categorical_dtype(rhs1.dtype) and rhs2.dtype == ty.uint32
        )

        task_id = (
            OpCode.BROADCAST_BINARY_OP
            if rhs1_scalar or rhs2_scalar
            else OpCode.BINARY_OP
        )
        plan = Map(runtime, task_id)

        # TODO: For now we don't have a case where the first op is scalar
        assert not rhs1_scalar

        plan.add_scalar_arg(runtime.get_binary_op_code(op).value, ty.int32)
        lhs.add_to_plan_output_only(plan)
        rhs1.add_to_plan(plan, True)
        rhs2.add_to_plan(plan, True)
        if rhs1_scalar or rhs2_scalar:
            plan.add_scalar_arg(rhs2_scalar, ty.bool)
        plan.execute(lhs.launch_domain)

        return lhs

    def fillna(self, rhs):
        assert self.nullable

        runtime = self.runtime

        lhs = _create_column(
            self.storage, self.dtype, ipart=self.primary_ipart, nullable=False
        )

        plan = Map(runtime, OpCode.BROADCAST_FILLNA)

        lhs.add_to_plan_output_only(plan)
        self.add_to_plan(plan, True)
        rhs.add_to_plan(plan)

        plan.execute(lhs.launch_domain)

        return lhs

    def equals(self, rhs, unwrap):
        rhs1, rhs2 = self, rhs

        assert rhs1.dtype == rhs2.dtype

        runtime = self.runtime
        plan = ScalarMap(runtime, OpCode.EQUALS, ty.bool)

        rhs1.add_to_plan(plan, True)
        rhs2.add_to_plan(plan, True)

        results = plan.execute(rhs1.launch_domain).get_futures(rhs1.num_pieces)

        plan = ScalarMap(runtime, OpCode.SCALAR_REDUCTION, ty.bool)

        plan.add_scalar_arg(ty.get_aggregation_op_id("all"), ty.int32)
        for result in results:
            plan.add_future(result)

        result = plan.execute_single()

        if unwrap:
            result = result.get_scalar().value

        return result

    def unary_op(self, op):
        runtime = self.runtime

        result = self.storage.create_column(
            self.dtype, ipart=self.primary_ipart, nullable=False
        )

        plan = Map(runtime, OpCode.UNARY_OP)

        plan.add_scalar_arg(runtime.get_unary_op_code(op), ty.int32)
        result.add_to_plan_output_only(plan)
        self.add_to_plan(plan, True)
        plan.execute(result.launch_domain)

        result.set_bitmask(self.bitmask)

        return result

    def isna(self):
        result = self.storage.create_column(
            ty.bool, ipart=self.primary_ipart, nullable=False
        )

        plan = Map(self.runtime, OpCode.ISNA)

        result.add_to_plan_output_only(plan)
        # TODO: We don't need to pass the whole column here but the bitmask
        self.add_to_plan(plan, True)
        plan.execute(result.launch_domain)

        return result

    def notna(self):
        result = self.storage.create_column(
            ty.bool, ipart=self.primary_ipart, nullable=False
        )

        plan = Map(self.runtime, OpCode.NOTNA)

        result.add_to_plan_output_only(plan)
        # TODO: We don't need to pass the whole column here but the bitmask
        self.add_to_plan(plan, True)
        plan.execute(result.launch_domain)

        return result

    def unary_reduction(self, op, skipna=True):
        runtime = self.runtime
        lhs_dtype = ty.get_reduction_result_type(op, self.dtype)

        components = None
        if self.num_pieces == 1 or op not in ["mean", "var", "std"]:
            plan = ScalarMap(runtime, OpCode.UNARY_REDUCTION, lhs_dtype)

            plan.add_scalar_arg(ty.get_aggregation_op_id(op), ty.int32)
            self.add_to_plan(plan, True)
            plan.add_dtype_arg(lhs_dtype)

            local_agg_fm = plan.execute(self.launch_domain)
            components = local_agg_fm.get_futures(self.num_pieces)

        elif op == "var" or op == "std":
            components = [
                self.unary_reduction("sqsum"),
                self.unary_reduction("sum"),
                self.unary_reduction("count"),
            ]

        else:
            assert op == "mean"
            components = [
                self.unary_reduction("sum"),
                self.unary_reduction("count"),
            ]

        if len(components) == 1:
            return components[0]

        plan = ScalarMap(runtime, OpCode.SCALAR_REDUCTION, lhs_dtype)

        op = "sum" if op in ["sqsum", "count"] else op
        plan.add_scalar_arg(ty.get_aggregation_op_id(op), ty.int32)
        for component in components:
            plan.add_future(component)

        return plan.execute_single()

    def copy_if_else(self, cond, other=None, negate=False):
        assert cond.dtype == ty.bool

        has_other = other is not None
        other_is_scalar = not isinstance(other, Column)

        if has_other:
            if other_is_scalar:
                other = self.runtime.create_scalar(other, self.dtype)
            elif other.dtype != self.dtype:
                other = other.astype(self.dtype)

        runtime = self.runtime

        plan = Map(runtime, OpCode.COPY_IF_ELSE)

        nullable = (
            self.nullable
            or not has_other
            or (not other_is_scalar and other.nullable)
        )
        lhs = self.storage.create_column(
            self.dtype, self.primary_ipart, nullable
        )
        if ty.is_string_dtype(self.dtype):
            offsets_storage = runtime.create_output_storage()
            chars_storage = runtime.create_output_storage()
            lhs.add_child(
                offsets_storage.create_column(ty.int32, nullable=False)
            )
            lhs.add_child(chars_storage.create_column(ty.int8, nullable=False))
            lhs = lhs.as_string_column()

        lhs.add_to_plan_output_only(plan)
        self.add_to_plan(plan, True)
        cond.add_to_plan(plan, True)
        plan.add_scalar_arg(negate, ty.bool)
        plan.add_scalar_arg(has_other, ty.bool)
        if has_other:
            plan.add_scalar_arg(other_is_scalar, ty.bool)
            other.add_to_plan(plan, True)
        plan.execute(self.launch_domain)

        return lhs

    def scan_op(self, op, skipna):
        result = self.storage.create_isomorphic_column(self)

        rt = self.runtime

        # Perform scan operation locally
        plan = Map(rt, OpCode.SCAN)

        # This is a local scan
        plan.add_scalar_arg(True, ty.bool)
        plan.add_scalar_arg(rt.get_scan_op_code(op), ty.int32)
        plan.add_scalar_arg(skipna, ty.bool)

        result.add_to_plan(plan, False)
        self.add_to_plan(plan, True)

        if self.num_pieces > 1:
            plan.add_scalar_arg(True, ty.bool)
            agg_buffer_storage = rt.create_storage(self.num_pieces)
            agg_buffer = agg_buffer_storage.create_column(
                self.dtype, nullable=self.nullable
            )
            agg_buffer.add_to_plan(plan, False)
            del agg_buffer_storage
        else:
            plan.add_scalar_arg(False, ty.bool)

        plan.execute(self.launch_domain)

        if self.num_pieces > 1:
            plan = Map(rt, OpCode.SCAN)

            plan.add_scalar_arg(False, ty.bool)
            plan.add_scalar_arg(rt.get_scan_op_code(op), ty.int32)
            plan.add_scalar_arg(skipna, ty.bool)

            result.add_to_plan(plan, None)
            agg_buffer.add_to_plan(plan, True, proj=None)

            plan.execute(result.launch_domain)

            del agg_buffer

        return result

    def cast_unsafe(self, result_dtype):
        assert self.dtype.itemsize == result_dtype.itemsize
        assert len(self.children) == 0

        return Column(
            self.runtime, self.data.cast_unsafe(result_dtype), self.bitmask
        )

    def materialize_indices(self, start, step):
        plan = Map(self.runtime, OpCode.MATERIALIZE)
        plan.add_future(start)
        plan.add_future(step)
        self.add_to_plan_output_only(plan)
        plan.execute(self.launch_domain)

    def export_offsets(self):
        rt = self.runtime
        rect = self.ispace.get_bounds()
        offset_size = rect.hi[0] - rect.lo[0] + 2
        offsets_ispace = rt.find_or_create_index_space(offset_size)
        offsets_storage = rt.create_storage(offsets_ispace)
        offsets = offsets_storage.create_column(ty.int32, nullable=False)

        plan = Map(rt, OpCode.EXPORT_OFFSETS)
        self.add_to_plan(plan, True)
        offsets.add_to_plan(plan, False)
        plan.execute_single()

        return offsets

    def all_to_ranges(self):
        children = [child.all_to_ranges() for child in self.children]
        if ty.is_string_dtype(self.dtype):
            children[0] = children[0].to_ranges(self, children[1])
        new_self = self.clone(shallow=True)
        new_self.set_primary_ipart(self.primary_ipart, recurse=False)
        new_self.children = children
        return new_self

    def to_ranges(self, parent, values):
        runtime = self.runtime

        result = parent.storage.create_column(
            ty.range64,
            ipart=parent.primary_ipart,
            nullable=False,
        )

        plan = Map(runtime, OpCode.OFFSETS_TO_RANGES)

        result.add_to_plan(plan, False)
        self.add_to_plan(plan, True)
        values.add_to_plan(plan, True)
        plan.execute(self.launch_domain)

        return result

    def all_to_offsets(self):
        children = [child.all_to_offsets() for child in self.children]
        if ty.is_string_dtype(self.dtype):
            children[0] = children[0].to_offsets()
        new_self = self.clone(shallow=True)
        new_self.set_primary_ipart(self.primary_ipart, recurse=False)
        new_self.children = children
        return new_self

    def to_offsets(self):
        runtime = self.runtime

        result_storage = runtime.create_output_storage()
        result = result_storage.create_column(ty.int32, nullable=False)

        plan = Map(runtime, OpCode.RANGES_TO_OFFSETS)

        result.add_to_plan_output_only(plan)
        self.add_to_plan(plan, True)

        counts = plan.execute(self.launch_domain)

        result_storage = plan.promote_output_storage(result_storage)
        runtime.register_external_weighted_partition(
            result_storage.default_ipart, counts
        )

        return result

    def repartition(self, num_pieces):
        if self.num_pieces == num_pieces:
            return self

        runtime = self.runtime

        new_self = self.all_to_ranges().clone()

        new_cspace = runtime.find_or_create_color_space(num_pieces)
        new_ipart = runtime.create_equal_partition(new_self.ispace, new_cspace)
        new_self.set_primary_ipart(new_ipart)

        new_self = new_self.all_to_offsets()

        return new_self

    def to_numpy(self):
        to_convert = self
        if self.has_nulls():
            dtype = ty.promote_dtype(to_convert.dtype)
            if dtype != to_convert.dtype:
                to_convert = to_convert.astype(dtype)
            null = self.runtime.create_scalar(ty.null_value(dtype), dtype)
            null = null.get_future()
            to_convert = to_convert.fillna(null)
        return to_convert.data.to_numpy()

    def to_pandas(self, schema_only=False):
        dtype = self.dtype.to_pandas()
        if schema_only:
            return pd.Series([], dtype=dtype)
        else:
            return pd.Series(self.to_numpy(), dtype=dtype, copy=True)

    def as_column(self):
        return Column(self.runtime, self.data, self.bitmask, self.children)

    def as_string_column(self):
        return StringColumn(
            self.runtime,
            self.data,
            self.bitmask,
            self.children,
        )

    def as_category_column(self):
        return CategoryColumn(
            self.runtime, self.data, self.bitmask, self.children
        )

    def as_replicated_column(self):
        return ReplicatedColumn(
            self.runtime, self.data, self.bitmask, self.children
        )


class CategoryColumn(Column):
    def __init__(self, runtime, data, bitmask, children):
        super(CategoryColumn, self).__init__(runtime, data, bitmask, children)

    def add_to_plan_output_only(self, plan):
        if type(self.children[1]) != ReplicatedColumn:
            new_self = self.as_column()
        else:
            new_self = Column(
                self.runtime, self.data, self.bitmask, [self.children[0]]
            )
        new_self.add_to_plan_output_only(plan)

    def initialize_bitmask(self):
        assert self.nullable

        rt = self.runtime

        plan_init = Map(rt, OpCode.INIT_BITMASK)

        code_dtype = self.children[0].dtype
        null_value = rt.create_scalar(ty.null_value(self.dtype), code_dtype)
        plan_init.add_future(null_value.get_future())

        bitmask = Column(rt, self.bitmask._storage)
        bitmask.add_to_plan_output_only(plan_init)

        self.children[0].add_to_plan(plan_init, True)

        plan_init.execute(self.launch_domain)

    def get_codes(self):
        assert ty.is_categorical_dtype(self.dtype)
        result = self.children[0].astype(ty.int32)
        result.bitmask = self.bitmask
        return result

    def astype(self, result_dtype, **kwargs):
        # TODO: Once we start to support primitive values as categories,
        #       we should perform more precise type checking here.
        if ty.is_primitive_dtype(result_dtype):
            raise ValueError("Cannot cast object dtype to {result_dtype}")

        return super(CategoryColumn, self).astype(result_dtype, **kwargs)

    def to_category_column(self, dtype=None):
        if isinstance(dtype, str):
            dtype = None

        if dtype is None:
            return self
        else:
            return self.astype(ty.string).to_category_column(dtype)

    def to_pandas(self, schema_only=False):
        dtype = self.dtype.to_pandas()
        if schema_only:
            return pd.Categorical.from_codes([], dtype=dtype)
        codes = self.get_codes()
        if self.nullable:
            null_value = self.runtime.create_scalar(-1, codes.dtype)
            null_value = null_value.get_future()
            codes = codes.fillna(null_value)
        return pd.Series(
            pd.Categorical.from_codes(codes.to_numpy(), dtype=dtype)
        )

    def to_numpy(self):
        return self.to_pandas().to_numpy()

    def fillna(self, rhs):
        assert self.nullable
        assert rhs.dtype is ty.string

        code = self.dtype.encode(rhs)

        code_column = Column(self.runtime, self.children[0].data, self.bitmask)
        filled = code_column.fillna(code)
        return CategoryColumn(
            self.runtime, self.data, None, [filled, self.children[1]]
        )


class StringColumn(Column):
    supported_binops = set(["eq", "gt", "ge", "lt", "le", "ne"])

    def __init__(self, runtime, column, bitmask, children):
        super(StringColumn, self).__init__(runtime, column, bitmask, children)

    @property
    def offsets(self):
        return self.children[0]

    @property
    def chars(self):
        return self.children[1]

    def unary_op(self, op_code, result_dtype=None):
        lhs = self.storage.create_column(
            self.dtype, ipart=self.primary_ipart, nullable=self.nullable
        )
        lhs_offsets = self.offsets.storage.create_column(
            self.offsets.dtype,
            ipart=self.offsets.primary_ipart,
            nullable=False,
        )
        lhs_chars = self.chars.storage.create_column(
            self.chars.dtype,
            ipart=self.chars.primary_ipart,
            nullable=False,
        )
        lhs.add_child(lhs_offsets)
        lhs.add_child(lhs_chars)

        lhs = lhs.as_string_column()

        runtime = self.runtime

        plan = Map(runtime, OpCode.STRING_UOP)
        plan.add_scalar_arg(op_code.value, ty.uint32)
        lhs.add_to_plan_output_only(plan)
        self.add_to_plan(plan, True)

        plan.execute(self.launch_domain)

        if self.nullable:
            lhs.set_bitmask(self.bitmask)

        return lhs

    def lower(self):
        return self.unary_op(StringMethods.LOWER)

    def upper(self):
        return self.unary_op(StringMethods.UPPER)

    def swapcase(self):
        return self.unary_op(StringMethods.SWAPCASE)

    def contains(self, pat):
        runtime = self.runtime

        lhs = self.storage.create_column(
            ty.bool,
            ipart=self.primary_ipart,
            nullable=False,
        )

        plan = Map(runtime, OpCode.CONTAINS)
        plan.add_scalar_arg(pat, ty.string)
        lhs.add_to_plan_output_only(plan)
        self.add_to_plan(plan, True)

        plan.execute(self.launch_domain)

        if self.nullable:
            lhs.set_bitmask(self.bitmask)

        return lhs

    def strip(self, to_strip):
        runtime = self.runtime

        lhs = self.storage.create_column(
            self.dtype,
            ipart=self.primary_ipart,
            nullable=False,
        )
        lhs_offsets = self.offsets.storage.create_column(
            self.offsets.dtype,
            ipart=self.offsets.primary_ipart,
            nullable=False,
        )
        chars_storage = self.runtime.create_output_storage()
        lhs_chars = chars_storage.create_column(
            self.chars.dtype,
            nullable=False,
        )
        lhs.add_child(lhs_offsets)
        lhs.add_child(lhs_chars)

        lhs = lhs.as_string_column()

        has_to_strip = to_strip is not None

        plan = Map(runtime, OpCode.STRIP)
        plan.add_scalar_arg(has_to_strip, ty.bool)
        if has_to_strip:
            plan.add_scalar_arg(to_strip, ty.string)
        lhs.add_to_plan_output_only(plan)
        self.add_to_plan(plan, True)

        plan.execute(self.launch_domain)

        if self.nullable:
            lhs.set_bitmask(self.bitmask)

        return lhs

    def zfill(self, width):
        runtime = self.runtime

        lhs = self.storage.create_column(
            self.dtype,
            ipart=self.primary_ipart,
            nullable=False,
        )
        lhs_offsets = self.offsets.storage.create_column(
            self.offsets.dtype,
            ipart=self.offsets.primary_ipart,
            nullable=False,
        )
        chars_storage = self.runtime.create_output_storage()
        lhs_chars = chars_storage.create_column(
            self.chars.dtype,
            nullable=False,
        )
        lhs.add_child(lhs_offsets)
        lhs.add_child(lhs_chars)

        lhs = lhs.as_string_column()

        plan = Map(runtime, OpCode.ZFILL)
        plan.add_scalar_arg(width, ty.int32)
        lhs.add_to_plan_output_only(plan)
        self.add_to_plan(plan, True)

        plan.execute(self.launch_domain)

        if self.nullable:
            lhs.set_bitmask(self.bitmask)

        return lhs

    def pad(self, width, side, fillchar):
        runtime = self.runtime

        lhs = self.storage.create_column(
            self.dtype,
            ipart=self.primary_ipart,
            nullable=False,
        )
        lhs_offsets = self.offsets.storage.create_column(
            self.offsets.dtype,
            ipart=self.offsets.primary_ipart,
            nullable=False,
        )
        chars_storage = self.runtime.create_output_storage()
        lhs_chars = chars_storage.create_column(
            self.chars.dtype,
            nullable=False,
        )
        lhs.add_child(lhs_offsets)
        lhs.add_child(lhs_chars)

        lhs = lhs.as_string_column()

        plan = Map(runtime, OpCode.PAD)
        plan.add_scalar_arg(width, ty.int32)
        plan.add_scalar_arg(runtime.get_pad_side_code(side), ty.int32)
        plan.add_scalar_arg(fillchar, ty.string)
        lhs.add_to_plan_output_only(plan)
        self.add_to_plan(plan, True)

        plan.execute(self.launch_domain)

        if self.nullable:
            lhs.set_bitmask(self.bitmask)

        return lhs

    def encode_category(self, category, can_fail=False):
        runtime = self.runtime

        plan = ScalarMap(runtime, OpCode.ENCODE_CATEGORY, ty.uint32)
        self.add_to_plan(plan, True)
        category.add_to_plan(plan)
        plan.add_scalar_arg(can_fail, ty.bool)

        return plan.execute_single()

    def to_datetime(self, format):
        runtime = self.runtime

        # FIXME: For now we always use datetime64[ns] type
        lhs = self.storage.create_column(
            ty.ts_ns,
            ipart=self.primary_ipart,
            nullable=False,
        )

        plan = Map(runtime, OpCode.TO_DATETIME)
        plan.add_scalar_arg(format, ty.string)
        lhs.add_to_plan_output_only(plan)
        self.add_to_plan(plan, True)

        plan.execute(self.launch_domain)

        if self.nullable:
            lhs.set_bitmask(self.bitmask)

        return lhs

    def to_category_column_cpu(self, dtype):
        rt = self.runtime

        nullable = dtype is not None or self.nullable
        if dtype is None:
            # Local de-duplication
            storage = rt.create_output_storage()
            result_column = storage.create_similar_column(self, nullable=False)

            plan = Map(rt, OpCode.DROP_DUPLICATES_CATEGORIES)
            plan.add_scalar_arg(1, ty.uint32)
            result_column.add_to_plan_output_only(plan)
            self.add_to_plan(plan, True)
            plan.execute(self.launch_domain)
            del plan

            radix = rt.radix
            num_pieces = result_column.num_pieces
            while num_pieces > 1:
                # Global de-duplication
                num_pieces = (num_pieces + radix - 1) // radix
                local_dedup_column = result_column

                storage = rt.create_output_storage()
                result_column = storage.create_similar_column(
                    self, nullable=False
                )

                plan = Map(rt, OpCode.DROP_DUPLICATES_CATEGORIES)
                plan.add_scalar_arg(radix, ty.uint32)
                result_column.add_to_plan_output_only(plan)
                for r in range(radix):
                    proj_id = rt.get_radix_functor_id(radix, r)
                    local_dedup_column.add_to_plan(plan, True, proj=proj_id)
                launch_domain = Rect([num_pieces])
                plan.execute(launch_domain)
                del plan

            categories_column = result_column.as_replicated_column()
            dtype = ty.CategoricalDtype(categories_column)

        encode_result = self.storage.create_column(
            dtype, ipart=self.primary_ipart, nullable=nullable
        )
        encode_result.add_child(
            self.storage.create_column(
                ty.uint32,
                ipart=self.primary_ipart,
                nullable=False,
            )
        )

        plan = Map(rt, OpCode.ENCODE)
        encode_result.add_to_plan_output_only(plan)
        dtype.categories_column.add_to_plan(plan, True)
        self.add_to_plan(plan, True)
        plan.execute(self.launch_domain)
        del plan

        encode_result.add_child(dtype.categories_column)
        return encode_result.as_category_column()

    def to_category_column_nccl(self):
        rt = self.runtime

        dict_column = rt.create_output_storage().create_column(
            self.dtype,
            nullable=False,
        )
        dict_offsets = rt.create_output_storage().create_column(
            self.offsets.dtype,
            nullable=False,
        )
        dict_chars = rt.create_output_storage().create_column(
            self.chars.dtype,
            nullable=False,
        )
        dict_column.add_child(dict_offsets)
        dict_column.add_child(dict_chars)

        result_dtype = ty.CategoricalDtype(dict_column.as_replicated_column())

        result_column = self.storage.create_column(
            result_dtype, ipart=self.primary_ipart, nullable=self.nullable
        )
        result_column.add_child(
            self.storage.create_column(
                ty.uint32,
                ipart=self.primary_ipart,
                nullable=False,
            )
        )
        plan = Map(rt, OpCode.ENCODE_NCCL)
        plan.add_scalar_arg(self.num_pieces, ty.uint32)
        result_column.add_to_plan_output_only(plan)
        dict_column.add_to_plan_output_only(plan)
        self.add_to_plan(plan, True)
        plan.add_future_map(rt._nccl_comm)
        rt.issue_fence()
        plan.execute(self.launch_domain)
        rt.issue_fence()
        del plan

        result_column.add_child(result_dtype.categories_column)
        return result_column.as_category_column()

    def to_category_column(self, dtype=None):
        if isinstance(dtype, str):
            dtype = None
        elif dtype is not None:
            dtype = ty.CategoricalDtype.from_pandas(self.runtime, dtype)

        if dtype is None and self.num_pieces > 1 and self.runtime.use_nccl:
            return self.to_category_column_nccl()
        else:
            return self.to_category_column_cpu(dtype)

    def to_numpy(self):
        return np.asarray(self.to_pandas().values)

    def to_pandas(self, schema_only=False):
        if schema_only:
            return pd.Series([], dtype=pd.StringDtype())

        # Recover a contiguous bitmask
        new_self = self.repartition(1).as_string_column()

        rect = new_self.offsets.ispace.get_bounds()
        num_elements = rect.hi[0] - rect.lo[0]

        if num_elements <= 0:
            return pd.Series([], dtype=pd.StringDtype())

        # Convert ranges back to offsets
        # XXX: We should keep this reference to the result of export_offsets
        #      to avoid it being collected
        offsets = new_self.offsets.data.to_raw_address()
        offsets_size = new_self.offsets.dtype.itemsize * (num_elements + 1)

        chars = new_self.chars.data.to_raw_address()
        char_rect = new_self.chars.ispace.get_bounds()
        char_size = char_rect.hi[0] - char_rect.lo[0] + 1

        bitmask_buf = None
        null_count = 0
        if new_self.nullable:
            null_count = new_self.bitmask.count_nulls().sum().get_value()
            if null_count > 0:
                bitmask = new_self.bitmask.compact_bitmask.to_raw_address()
                bitmask_size = (num_elements + 7) // 8
                bitmask_buf = pa.foreign_buffer(bitmask, bitmask_size)

        offsets_buf = pa.foreign_buffer(offsets, offsets_size)
        chars_buf = pa.foreign_buffer(chars, char_size)

        array = pa.StringArray.from_buffers(
            num_elements,
            offsets_buf,
            chars_buf,
            bitmask_buf,
            null_count,
        )

        return array.to_pandas().astype(pd.StringDtype())


class ReplicatedColumn(Column):
    def __init__(self, runtime, data, bitmask, children):
        super(ReplicatedColumn, self).__init__(
            runtime, data, bitmask, children
        )

    @property
    def partitioned(self):
        return False

    # Replicated columns ignore all partitioning related calls
    def set_primary_ipart(self, ipart, recurse=True):
        return self

    def all_to_ranges(self):
        return self

    def all_to_offsets(self):
        return self

    def add_to_plan(self, plan, read, proj=0):
        # Replicate columns are always read only
        super(ReplicatedColumn, self).add_to_plan(plan, True, None)

    def to_pandas(self, schema_only=False):
        if ty.is_string_dtype(self.dtype):
            return self.as_string_column().to_pandas(schema_only)
        else:
            raise ValueError("Unsupported dtype %s" % self.dtype)

    def equals(self, rhs, unwrap):
        runtime = self.runtime
        plan = ScalarMap(runtime, OpCode.EQUALS, ty.bool)

        self.add_to_plan(plan, True)
        rhs.add_to_plan(plan, True)

        result = plan.execute_single()
        if unwrap:
            result = result.get_scalar().value
        return result

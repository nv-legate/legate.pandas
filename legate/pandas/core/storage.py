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

from collections import OrderedDict

from legate.core import (
    Attach,
    Copy,
    Detach,
    FieldID,
    IndexFill,
    InlineMapping,
    Rect,
    Region,
    ffi,
    legion,
)

from legate.pandas.common import types as ty

from .bitmask import Bitmask
from .column import Column

try:
    long  # Python 2
except NameError:
    long = int  # Python 3

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


class ArrowAttach(Attach):
    def __init__(
        self,
        region,
        field,
        arrow_array,
        mapper=0,
        tag=0,
    ):
        self.launcher = legion.legion_attach_launcher_create(
            region.handle,
            region.handle,
            2,  # External array instance, from legion_config.h
        )
        self._launcher = ffi.gc(
            self.launcher, legion.legion_attach_launcher_destroy
        )
        legion.legion_attach_launcher_add_cpu_soa_field(
            self.launcher,
            ffi.cast("legion_field_id_t", field),
            ffi.cast("void*", arrow_array.address),
            True,
        )


class LegateStorageAdaptor(object):
    def __init__(self, storage):
        """
        Wrapper class for StorageSlice that implements the LegateStore methods
        """
        self._storage = storage

    @property
    def type(self):
        return self._storage.dtype.to_arrow()

    @property
    def kind(self):
        return (Region, FieldID)

    @property
    def storage(self):
        field_id = FieldID(
            self._storage.region.field_space, self._storage.field_id, self.type
        )
        return (self._storage.region, field_id)


class StorageSlice(object):
    """
    Proxy to a single field in a Storage
    """

    def __init__(self, runtime, storage, offset, region=None):
        self.offset = offset
        self.storage = storage

        self._runtime = runtime

        self._region = region
        self._field_id = None
        self._dtype = None

        self._primary_ipart = None
        self._num_pieces = None

    def __del__(self):
        self.storage.release_slice(self)

    @property
    def legate_store(self):
        return LegateStorageAdaptor(self)

    @property
    def fixed(self):
        return self.storage.fixed

    @property
    def region(self):
        if self._region is None:
            self._region = self.storage.get_region(self.offset)
        return self._region

    @property
    def fspace(self):
        return self.storage.fspace

    @property
    def field_id(self):
        if self._field_id is None:
            self._field_id = self.storage.get_field_id(self.offset)
        return self._field_id

    @property
    def physical_region(self):
        return self.storage.get_physical_region(self.offset)

    @property
    def ispace(self):
        return self.region.index_space

    @property
    def dtype(self):
        if self._dtype is None:
            self._dtype = self.storage.get_dtype(self.offset)
        return self._dtype

    @property
    def primary_ipart(self):
        assert self._primary_ipart is not None
        return self._primary_ipart

    @property
    def num_pieces(self):
        if self._num_pieces is None:
            bounds = self.primary_ipart.color_space.get_bounds()
            self._num_pieces = bounds.hi[0] - bounds.lo[0] + 1
        return self._num_pieces

    @property
    def launch_domain(self):
        return Rect([self.num_pieces])

    @property
    def cspace(self):
        return self.primary_ipart.color_space

    # This clones a slice with the primary partition unset
    def clone(self):
        return self.storage[self.offset]

    def set_primary_ipart(self, ipart):
        assert self._primary_ipart is None
        self._primary_ipart = ipart

    @property
    def has_primary_ipart(self):
        return self._primary_ipart is not None

    def get_view(self, ipart):
        return self._runtime.create_region_partition(self.region, ipart)

    def from_numpy(self, array):
        if ty.is_timestamp_dtype(self.dtype):
            array = array.view(self.dtype.storage_dtype.to_pandas())

        # 1. Make a temporary region that has the same index space
        # as the target and a field space with only one field.
        src_fspace = self._runtime.create_field_space()
        src_field_id = src_fspace.allocate_field(self.dtype)
        src_region = self._runtime.create_logical_region(
            self.region.index_space, src_fspace
        )

        # 2. Attach the array data to the temporary region
        attach = Attach(src_region, src_field_id, array, read_only=True)
        attach.set_restricted(False)
        src_physical_region = self._runtime.dispatch(attach)

        # 3. Issue a copy
        copy = Copy()
        copy.add_src_requirement(src_region, src_field_id)
        copy.add_dst_requirement(self.region, self.field_id)
        self._runtime.dispatch(copy)

        # 4. Finally, detach the Python array as we no longer need it
        detach = Detach(src_physical_region)
        self._runtime.dispatch(detach, unordered=True)

    def from_arrow(self, arrow_array):
        # 1. Make a temporary region that has the same index space
        # as the target and a field space with only one field.
        src_fspace = self._runtime.create_field_space()
        src_field_id = src_fspace.allocate_field(self.dtype)
        src_region = self._runtime.create_logical_region(
            self.region.index_space, src_fspace
        )

        # 2. Attach the array data to the temporary region
        attach = ArrowAttach(src_region, src_field_id, arrow_array)
        attach.set_restricted(False)
        src_physical_region = self._runtime.dispatch(attach)

        # 3. Issue a copy
        copy = Copy()
        copy.add_src_requirement(src_region, src_field_id)
        copy.add_dst_requirement(self.region, self.field_id)
        self._runtime.dispatch(copy)

        # 4. Finally, detach the Python array as we no longer need it
        detach = Detach(src_physical_region)
        self._runtime.dispatch(detach, unordered=True)

    def _to_raw_address(self):
        physical_region = self.physical_region
        bounds = self.ispace.get_bounds()
        dim = bounds.dim
        accessor = getattr(
            legion, "legion_physical_region_get_field_accessor_array_%dd" % dim
        )(
            physical_region.handle,
            ffi.cast("legion_field_id_t", self.field_id),
        )
        rect = ffi.new("legion_rect_%dd_t *" % dim)
        for i in xrange(dim):
            rect[0].lo.x[i] = bounds.lo[i]
            rect[0].hi.x[i] = bounds.hi[i]
        subrect = ffi.new("legion_rect_%dd_t *" % dim)
        offsets = ffi.new("legion_byte_offset_t[]", dim)
        base_ptr = getattr(
            legion, "legion_accessor_array_%dd_raw_rect_ptr" % dim
        )(accessor, rect[0], subrect, offsets)
        assert base_ptr is not None
        for i in xrange(dim):
            assert rect[0].lo.x[i] == subrect[0].lo.x[i]
            assert rect[0].hi.x[i] == subrect[0].hi.x[i]
        shape = tuple(
            [rect[0].hi.x[i] - rect[0].lo.x[i] + 1 for i in xrange(dim)]
        )
        strides = tuple([offsets[i].offset for i in xrange(dim)])
        base_ptr = long(ffi.cast("size_t", base_ptr))
        return (shape, strides, base_ptr)

    def to_raw_address(self):
        (_, _, base_ptr) = self._to_raw_address()

        return base_ptr

    def to_numpy(self, dtype=None):
        import numpy as np

        (shape, strides, base_ptr) = self._to_raw_address()

        if dtype is None:
            dtype = self.dtype.to_pandas()

        if base_ptr == 0:
            return np.array([], dtype=dtype)
        else:
            return np.asarray(
                _DummyArray(shape, dtype, base_ptr, strides, False)
            )

    def fill(self, value):
        # TODO: We need to convert the fill value to a storage dtype
        #       when the storage dtype is different from the column dtype.
        value = self._runtime.create_future(value, self.dtype)

        ipart = self.primary_ipart
        fill = IndexFill(
            self._runtime.get_partition(self.region, ipart),
            0,
            self.region,
            self.field_id,
            value,
            mapper=self._runtime.mapper_id,
        )
        self._runtime.dispatch(fill)

    def _unsafe_set_dtype(self, dtype):
        self._dtype = dtype

    def cast_unsafe(self, result_dtype):
        result = self.storage[self.offset]
        result._unsafe_set_dtype(result_dtype)
        if self.has_primary_ipart:
            result.set_primary_ipart(self.primary_ipart)
        return result


class Storage(object):
    """
    Data storage for Table

    One Storage corresponds to one logical region with one or more
    fields. It is Table's responsibility to know which of the fields
    the frame needs to access. One Storage can be shared by multiple
    Table objects.
    """

    def __init__(
        self,
        runtime,
        ispace=None,
        region=None,
        ipart=None,
        external=False,
        start_field_id=None,
    ):
        assert ispace is None or region is None
        assert ispace is not None or region is not None

        if region is None:
            self.ispace = ispace
            self._regions = []
            self._allocable = []
        else:
            self.ispace = region.index_space
            self._regions = [region]
            self._allocable = [not external]

        if start_field_id is None:
            start_field_id = runtime.pandas_field_id_base

        self._runtime = runtime
        self._physical_regions = {}

        self._field_ids = []
        self._field_id_to_offset = OrderedDict()
        self._region_indices = []
        self._dtypes = []

        self._ref_counts = {}
        self._free_lists = OrderedDict()
        self._next_field_id = start_field_id
        self._default_ipart = ipart

    @property
    def fixed(self):
        return True

    @property
    def empty(self):
        num_free_fields = sum([len(ls) for ls in self._free_lists.values()])
        return len(self._field_ids) - num_free_fields == 0

    def __getitem__(self, offset):
        assert 0 <= offset and offset < len(self._field_ids)
        self._inc_ref_count(offset)
        return StorageSlice(self._runtime, self, offset)

    def __setitem__(self, offset, value):
        raise TypeError("Slices cannot be assigned")

    def __del__(self):
        for region, allocable in zip(self._regions, self._allocable):
            if allocable:
                region.destroy(unordered=True)
        if self._runtime.trace_storages:
            self._storages.remove(self)
            self._runtime.dump_storage_stat()

    def _inc_ref_count(self, offset):
        if offset not in self._ref_counts:
            count = 0
        else:
            count = self._ref_counts[offset]
        count += 1
        self._ref_counts[offset] = count

    def _dec_ref_count(self, offset):
        assert offset in self._ref_counts
        self._ref_counts[offset] -= 1

    def _is_collectable(self, offset):
        return (
            self._allocable[self._region_indices[offset]]
            and offset in self._ref_counts
            and self._ref_counts[offset] == 0
        )

    def _add_to_freelist(self, slice):
        itemsize = slice.dtype.itemsize
        if itemsize not in self._free_lists:
            self._free_lists[itemsize] = list()
        self._free_lists[itemsize].append(slice.offset)
        self.unmap_physical_region(slice.offset)
        if self._runtime.trace_storages:
            self._runtime.dump_storage_stat()

    def _pop_freelist(self, itemsize):
        if itemsize not in self._free_lists:
            return None
        free_list = self._free_lists[itemsize]
        offset = free_list.pop()
        if len(free_list) == 0:
            del self._free_lists[itemsize]
        return offset

    def _get_next_field_id(self):
        field_id = self._next_field_id
        self._next_field_id += 1
        return field_id

    @property
    def region(self):
        return self._regions[-1]

    @property
    def fspace(self):
        raise ValueError(
            "Accessing field space of a normal storage is not allowed"
        )

    @property
    def current_region_index(self):
        return len(self._regions) - 1

    @property
    def has_space(self):
        return (
            len(self._regions) > 0
            and self._allocable[-1]
            and self.region.field_space.has_space
        )

    def _create_new_region(self):
        fspace = self._runtime.create_field_space()
        region = self._runtime.create_logical_region(self.ispace, fspace)
        self._regions.append(region)
        self._allocable.append(True)

    @property
    def default_ipart(self):
        if self._default_ipart is None:
            self._default_ipart = self._runtime.create_equal_partition(
                self.ispace,
                self._runtime.find_or_create_color_space(
                    self._runtime.num_pieces
                ),
            )
        return self._default_ipart

    def get_physical_region(self, offset):
        if offset in self._physical_regions:
            return self._physical_regions[offset]

        mapping = InlineMapping(
            self.get_region(offset), self.get_field_id(offset)
        )
        physical_region = self._runtime.dispatch(mapping)
        # Wait until it is valid before returning
        physical_region.wait_until_valid()

        self._physical_regions[offset] = physical_region
        return physical_region

    def unmap_physical_region(self, offset):
        if offset not in self._physical_regions:
            return
        self._runtime.unmap_physical_region(self._physical_regions[offset])
        self._physical_regions[offset] = None
        del self._physical_regions[offset]

    def get_field_id(self, offset):
        return self._field_ids[offset]

    def get_region(self, offset):
        return self._regions[self._region_indices[offset]]

    def get_dtype(self, offset):
        return self._dtypes[offset]

    def create_new_field(self, dtype):
        assert ty.is_legate_pandas_dtype(dtype)
        alloc_dtype = dtype.storage_dtype
        offset = self._pop_freelist(alloc_dtype.itemsize)
        if offset is not None:
            self._dtypes[offset] = dtype
            return self[offset]
        else:
            field_id = self._get_next_field_id()
            if not self.has_space:
                self._create_new_region()
            self.region.field_space.allocate_field(alloc_dtype, field_id)
            offset = len(self._field_ids)
            self._dtypes.append(dtype)
            self._field_ids.append(field_id)
            self._field_id_to_offset[field_id] = offset
            self._region_indices.append(self.current_region_index)
            if self._runtime.trace_storages:
                self._runtime.dump_storage_stat()
            return self[offset]

    def create_new_fields(self, dtypes):
        return [self.create_new_field(dtype) for dtype in dtypes]

    def create_column(
        self, dtype, ipart=None, nullable=True, bitmask=None, ctor=Column
    ):
        assert bitmask is None or nullable

        data = self.create_new_field(dtype)
        if nullable and bitmask is None:
            bitmask = self.create_new_field(Bitmask.alloc_type)
            bitmask = Bitmask(self._runtime, bitmask)
        elif not nullable:
            bitmask = None

        column = ctor(self._runtime, data, bitmask, [])
        if ipart is None:
            ipart = self.default_ipart
        column.set_primary_ipart(ipart, recurse=False)
        return column

    def create_columns(self, dtypes, ipart=None, nullable=True):
        if type(nullable) != list:
            nullable = [nullable] * len(dtypes)
        return [
            self.create_column(dtype, ipart, nullable)
            for dtype, nullable in zip(dtypes, nullable)
        ]

    def create_isomorphic_column(self, to_copy, dedup_bitmask=False):
        if not to_copy.partitioned:
            return to_copy
        result = to_copy.storage.create_column(
            to_copy.dtype,
            to_copy.primary_ipart,
            to_copy.nullable,
            bitmask=to_copy.bitmask if dedup_bitmask else None,
            ctor=type(to_copy),
        )
        result.children = [
            self.create_isomorphic_column(child, dedup_bitmask)
            for child in to_copy.children
        ]
        return result

    def create_isomorphic_columns(self, columns):
        return [self.create_isomorphic_column(column) for column in columns]

    def create_similar_column(self, to_imitate, nullable=None):
        if not to_imitate.partitioned:
            return to_imitate
        result = self.create_column(
            to_imitate.dtype,
            nullable=to_imitate.nullable if nullable is None else nullable,
            ctor=type(to_imitate),
        )
        children_to_copy = to_imitate.children
        if ty.is_string_dtype(to_imitate.dtype):
            storages = [
                self._runtime.create_output_storage() for _ in children_to_copy
            ]
        else:
            storages = [self] * len(children_to_copy)
        result.children = [
            storage.create_similar_column(child, nullable)
            for storage, child in zip(storages, children_to_copy)
        ]
        return result

    def import_field(self, region, field_id, dtype):
        assert self._regions[0] == region

        if field_id not in self._field_id_to_offset:
            offset = len(self._field_ids)
            self._field_ids.append(field_id)
            self._field_id_to_offset[field_id] = offset
            self._region_indices.append(0)
            self._dtypes.append(dtype)
        else:
            offset = self._field_id_to_offset[field_id]
            assert dtype == self._dtypes[offset]

        return self[offset]

    def import_column(
        self, region, field_id, dtype, ipart=None, nullable=True
    ):
        assert self._regions[0] == region

        data = self.import_field(region, field_id, dtype)
        bitmask = None
        if nullable:
            bitmask = self.create_new_field(Bitmask.alloc_type)
            bitmask = Bitmask(self._runtime, bitmask)

        column = Column(self._runtime, data, bitmask)
        if ipart is None:
            ipart = self.default_ipart
        column.set_primary_ipart(ipart)
        return column

    # NOTE: This method must be used only when promoting an external storage
    #       to a normal storage
    def _import_slice(self, slice):
        assert slice.offset == len(self._field_ids)
        self._inc_ref_count(slice.offset)
        self.import_field(self._regions[0], slice.field_id, slice.dtype)

    def release_slice(self, slice):
        self._dec_ref_count(slice.offset)
        if self._is_collectable(slice.offset):
            self._add_to_freelist(slice)

    def dump_stat(self):
        volume = self.ispace.get_volume()
        num_fields = len(self._field_ids)
        num_free_fields = sum([len(ls) for ls in self._free_lists.values()])
        total_bytes = sum([volume * dtype.itemsize for dtype in self._dtypes])
        if total_bytes == 0:
            return 0, 0
        free_stats = []
        total_free_bytes = 0
        for itemsize, l in self._free_lists.items():
            free_bytes = volume * itemsize * len(l)
            free_stats.append(
                "        itemsize %d: %d free fields, %.2f MBs"
                % (itemsize, len(l), free_bytes / 1000000.0)
            )
            total_free_bytes += free_bytes
        stat = (
            "    - [Storage %s] %d elements, %d fields allocated (%.2f MBs), "
            "%d fields unused (%.2f MBs), %f %% used "
            % (
                str(self),
                volume,
                num_fields,
                total_bytes / 1000000.0,
                num_free_fields,
                total_free_bytes / 1000000.0,
                (total_bytes - total_free_bytes) / total_bytes * 100,
            )
        )
        if len(free_stats) > 0:
            stat += "\n      free lists:"
            for free_stat in free_stats:
                stat += "\n" + free_stat
        print(stat)
        return total_bytes, total_free_bytes


class OutputStorage(object):
    """
    Wrapper for output regions

    """

    def __init__(self, runtime):
        self.fspace = runtime.create_field_space()

        self._runtime = runtime
        self._next_field_id = runtime.pandas_field_id_base
        self._field_ids = []
        self._dtypes = []
        self._slices = []

    @property
    def fixed(self):
        return False

    def _get_next_field_id(self):
        field_id = self._next_field_id
        self._next_field_id += 1
        return field_id

    def release_slice(self, slice):
        pass

    def get_region(self, offset):
        raise ValueError("Accessing region of an output slice is not allowed")

    def get_physical_region(self, offset):
        raise ValueError(
            "Accessing physical region of an output slice is not allowed"
        )

    def get_field_id(self, offset):
        return self._field_ids[offset]

    def get_dtype(self, offset):
        return self._dtypes[offset]

    def create_new_field(self, dtype):
        alloc_dtype = dtype.storage_dtype
        field_id = self._get_next_field_id()
        self.fspace.allocate_field(alloc_dtype, field_id)
        offset = len(self._field_ids)
        slice = StorageSlice(self._runtime, self, offset)
        self._field_ids.append(field_id)
        self._dtypes.append(dtype)
        self._slices.append(slice)
        return slice

    def create_bitmask(self):
        bitmask = self.create_new_field(Bitmask.alloc_type)
        return Bitmask(self._runtime, bitmask)

    def create_column(self, dtype, nullable=True, ctor=Column):
        data = self.create_new_field(dtype)
        if nullable:
            bitmask = self.create_bitmask()
        elif not nullable:
            bitmask = None
        return ctor(self._runtime, data, bitmask, [])

    def create_columns(self, dtypes, nullable=True):
        if type(nullable) != list:
            nullable = [nullable] * len(dtypes)
        return [
            self.create_column(dtype, nullable)
            for dtype, nullable in zip(dtypes, nullable)
        ]

    def create_similar_column(self, to_imitate, nullable=None):
        if not to_imitate.partitioned:
            return to_imitate
        result = self.create_column(
            to_imitate.dtype,
            to_imitate.nullable if nullable is None else nullable,
            type(to_imitate),
        )
        children_to_copy = to_imitate.children
        if ty.is_string_dtype(to_imitate.dtype):
            storages = [
                self._runtime.create_output_storage() for _ in children_to_copy
            ]
        else:
            storages = [self] * len(children_to_copy)
        result.children = [
            storage.create_similar_column(child, nullable)
            for storage, child in zip(storages, children_to_copy)
        ]
        return result

    def create_similar_columns(self, columns):
        return [self.create_similar_column(column) for column in columns]

    def promote(self, region, ipart):
        new_storage = Storage(
            self._runtime,
            region=region,
            ipart=ipart,
            start_field_id=self._next_field_id,
        )
        for slice in self._slices:
            slice.set_primary_ipart(ipart)
            new_storage._import_slice(slice)
            slice.storage = new_storage
        if self._runtime.trace_storages:
            self._runtime._storages.append(new_storage)
            self._runtime.dump_storage_stat()
        return new_storage


class _DummyArray(object):
    __slots__ = ["__array_interface__"]

    def __init__(self, shape, field_type, base_ptr, strides, read_only):
        # See: https://docs.scipy.org/doc/numpy/reference/arrays.interface.html
        self.__array_interface__ = {
            "version": 3,
            "shape": shape,
            "typestr": field_type.str,
            "data": (base_ptr, read_only),
            "strides": strides,
        }

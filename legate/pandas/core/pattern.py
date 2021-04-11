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
from enum import IntEnum

from legate.core import ArgumentMap, BufferBuilder, IndexTask, Task

from legate.pandas.common import types as ty

NOT_IMPLEMENTED_MESSAGE = "Need Legate implementation"


class PandasBufferBuilder(BufferBuilder):
    def pack_dtype(self, dtype):
        self.pack_32bit_int(ty.encode_dtype(dtype))


class Permission(IntEnum):
    NO_ACCESS = 0
    READ = 1
    WRITE = 2
    READ_WRITE = 3


class ScalarArg(object):
    _serializers = {
        ty.bool: BufferBuilder.pack_bool,
        ty.int8: BufferBuilder.pack_8bit_int,
        ty.int16: BufferBuilder.pack_16bit_int,
        ty.int32: BufferBuilder.pack_32bit_int,
        ty.int64: BufferBuilder.pack_64bit_int,
        ty.uint8: BufferBuilder.pack_8bit_uint,
        ty.uint16: BufferBuilder.pack_16bit_uint,
        ty.uint32: BufferBuilder.pack_32bit_uint,
        ty.uint64: BufferBuilder.pack_64bit_uint,
        ty.float32: BufferBuilder.pack_32bit_float,
        ty.float64: BufferBuilder.pack_64bit_float,
        ty.string: BufferBuilder.pack_string,
    }

    def __init__(self, value, dtype):
        self._value = value
        self._dtype = dtype

    def pack(self, buf):
        if self._dtype in self._serializers:
            self._serializers[self._dtype](buf, self._value)
        else:
            raise ValueError("Unsupported data type: %s" % str(self._dtype))


class DtypeArg(object):
    def __init__(self, dtype):
        self._dtype = dtype

    def pack(self, buf):
        buf.pack_dtype(self._dtype)


class AccessorArg(object):
    def __init__(self, region_idx, field_id, dtype):
        self._region_idx = region_idx
        self._field_id = field_id
        self._dtype = dtype

    def pack(self, buf):
        buf.pack_dtype(self._dtype)
        buf.pack_32bit_uint(self._region_idx)
        buf.pack_accessor(self._field_id)


_single_task_calls = {
    Permission.NO_ACCESS: Task.add_no_access_requirement,
    Permission.READ: Task.add_read_requirement,
    Permission.WRITE: Task.add_write_requirement,
    Permission.READ_WRITE: Task.add_read_write_requirement,
}

_index_task_calls = {
    Permission.NO_ACCESS: IndexTask.add_no_access_requirement,
    Permission.READ: IndexTask.add_read_requirement,
    Permission.WRITE: IndexTask.add_write_requirement,
    Permission.READ_WRITE: IndexTask.add_read_write_requirement,
}


class _Broadcast(object):
    def add(self, runtime, task, arg, fields):
        f = _index_task_calls[arg.permission]
        f(task, arg.region, fields, 0, parent=arg.region, tag=arg.tag)

    def add_single(self, task, arg, fields):
        f = _single_task_calls[arg.permission]
        f(task, arg.region, fields, tag=arg.tag, flags=arg.flags)

    def __hash__(self):
        return hash("Broadcast")


Broadcast = _Broadcast()


class Projection(object):
    def __init__(self, ipart, proj=0):
        self.ipart = ipart
        self.proj = proj

    def add(self, runtime, task, arg, fields):
        part = runtime.get_partition(arg.region, self.ipart)
        f = _index_task_calls[arg.permission]
        f(task, part, fields, self.proj, tag=arg.tag, flags=arg.flags)

    def add_single(self, task, arg, fields):
        f = _single_task_calls[arg.permission]
        f(task, arg.region, fields, tag=arg.tag)

    def __hash__(self):
        return hash((self.ipart, self.proj))


class RegionArg(object):
    def __init__(self, region, proj, permission, tag, flags):
        self.region = region
        self.proj = proj
        self.permission = permission
        self.tag = tag
        self.flags = flags

    def __repr__(self):
        return (
            str(self.region)
            + ","
            + str(self.proj)
            + ","
            + str(self.permission)
            + ","
            + str(self.tag)
            + ","
            + str(self.flags)
        )

    def __hash__(self):
        return hash(
            (self.region, self.proj, self.permission, self.tag, self.flags)
        )

    def __eq__(self, other):
        return (
            self.region == other.region
            and self.proj == other.proj
            and self.permission == other.permission
            and self.tag == other.tag
            and self.flags == other.flags
        )


class Map(object):
    def __init__(self, runtime, task_id, tag=0):
        assert type(tag) != bool
        self._runtime = runtime
        self._task_id = runtime.get_task_id(task_id)
        self._args = list()
        self._region_args = OrderedDict()
        self._output_region_args = OrderedDict()
        self._output_reqs = OrderedDict()
        self._promoted_storages = OrderedDict()
        self._next_region_idx = 0
        self._next_output_region_idx = 0
        self._projections = list()
        self._future_args = list()
        self._future_map_args = list()
        self._tag = tag

    def __del__(self):
        self._region_args.clear()
        self._output_region_args.clear()
        self._output_reqs.clear()
        self._promoted_storages.clear()
        self._projections.clear()
        self._future_args.clear()
        self._future_map_args.clear()

    def _add_region_arg(self, region_arg, field_id):
        if region_arg not in self._region_args:
            idx = self._next_region_idx
            self._next_region_idx += 1
            self._region_args[region_arg] = ([field_id], idx)
            return idx
        else:
            (fields, idx) = self._region_args[region_arg]
            if field_id not in fields:
                fields.append(field_id)
            return idx

    def _add_output_region_arg(self, storage, ipart, field_id):
        if (storage, ipart) not in self._output_region_args:
            idx = self._next_output_region_idx
            self._next_output_region_idx += 1
            self._output_region_args[(storage, ipart)] = (set([field_id]), idx)
            return idx
        else:
            (fields, idx) = self._output_region_args[(storage, ipart)]
            if field_id not in fields:
                fields.add(field_id)
            return idx

    def add_scalar_arg(self, value, dtype):
        self._args.append(ScalarArg(value, dtype))

    def add_dtype_arg(self, dtype):
        self._args.append(DtypeArg(dtype))

    def add_region_arg(self, storage, proj, perm, tag, flags):
        # For empty fields we override the permission with no access
        # to avoid bogus uninitilaized access warnings from the runtime
        if storage.dtype.itemsize == 0:
            perm = Permission.NO_ACCESS
        region_idx = self._add_region_arg(
            RegionArg(storage.region, proj, perm, tag, flags),
            storage.field_id,
        )

        self._args.append(
            AccessorArg(
                region_idx,
                storage.field_id,
                storage.dtype,
            )
        )

    def add_no_access(self, storage, proj, tag=0, flags=0):
        self.add_region_arg(storage, proj, Permission.NO_ACCESS, tag, flags)

    def add_input(self, storage, proj, tag=0, flags=0):
        self.add_region_arg(storage, proj, Permission.READ, tag, flags)

    def add_output(self, storage, proj, tag=0, flags=0):
        self.add_region_arg(storage, proj, Permission.WRITE, tag, flags)

    def add_inout(self, storage, proj, tag=0, flags=0):
        self.add_region_arg(storage, proj, Permission.READ_WRITE, tag, flags)

    def add_output_only(self, storage):
        ipart = storage.primary_ipart if storage.storage.fixed else None
        output_region_idx = self._add_output_region_arg(
            storage.storage, ipart, storage.field_id
        )
        self._args.append(
            AccessorArg(
                output_region_idx,
                storage.field_id,
                storage.dtype,
            )
        )

    def add_future(self, future):
        self._future_args.append(future)

    def add_future_map(self, future_map):
        self._future_map_args.append(future_map)

    def build_task(self, launch_domain, argbuf):
        for arg in self._args:
            arg.pack(argbuf)
        task = IndexTask(
            self._task_id,
            launch_domain,
            self._runtime.empty_argmap,
            argbuf.get_string(),
            argbuf.get_size(),
            mapper=self._runtime.mapper_id,
            tag=self._tag,
        )

        for region_arg, (fields, _) in self._region_args.items():
            region_arg.proj.add(self._runtime, task, region_arg, fields)
        for future in self._future_args:
            task.add_future(future)
        for future_map in self._future_map_args:
            task.add_point_future(ArgumentMap(future_map=future_map))
        for (storage, ipart), (fields, _) in self._output_region_args.items():
            output_req = self._runtime.create_output_region(
                storage, list(fields), ipart=ipart
            )
            if not storage.fixed:
                self._output_reqs[storage] = output_req
            task.add_output(output_req)
        return task

    def build_single_task(self, argbuf):
        assert len(self._output_region_args) == 0
        for arg in self._args:
            arg.pack(argbuf)
        task = Task(
            self._task_id,
            argbuf.get_string(),
            argbuf.get_size(),
            mapper=self._runtime.mapper_id,
            tag=self._tag,
        )
        for region_arg, (fields, _) in self._region_args.items():
            region_arg.proj.add_single(task, region_arg, fields)
        for future in self._future_args:
            task.add_future(future)
        if len(self._region_args) == 0:
            task.set_local_function(True)
        return task

    def execute(self, launch_domain):
        # Note that we should hold a reference to this buffer
        # until we launch a task, otherwise the Python GC will
        # collect the Python object holding the buffer, which
        # in turn will deallocate the C side buffer.
        argbuf = PandasBufferBuilder()
        task = self.build_task(launch_domain, argbuf)
        result = self._runtime.dispatch(task)
        for storage, output_req in self._output_reqs.items():
            region = output_req.get_region()
            ipart = output_req.get_partition().index_partition
            self._promoted_storages[storage] = storage.promote(region, ipart)
        return result

    def execute_single(self):
        argbuf = PandasBufferBuilder()
        return self._runtime.dispatch(self.build_single_task(argbuf))

    def promote_output_storage(self, storage):
        assert not storage.fixed
        assert storage in self._promoted_storages
        return self._promoted_storages[storage]


class ScalarMap(Map):
    def __init__(self, runtime, task_id, lhs_dtype, tag=0):
        super(ScalarMap, self).__init__(runtime, task_id, tag)
        self._lhs_dtype = lhs_dtype

    def add_output(self, region, field_id, dtype, proj):
        raise ValueError("ScalarMap cannot take an output region")

    def add_inout(self, region, field_id, dtype, proj):
        raise ValueError("ScalarMap cannot take an inout region")

    def execute(self, launch_domain):
        argbuf = PandasBufferBuilder()
        return self._runtime.dispatch(
            self.build_task(launch_domain, argbuf)
        ).cast(self._lhs_dtype)

    def execute_single(self):
        argbuf = PandasBufferBuilder()
        return self._runtime.dispatch(self.build_single_task(argbuf)).cast(
            self._lhs_dtype
        )


class ScalarReduce(ScalarMap):
    def __init__(self, runtime, task_id, lhs_dtype, tag=0):
        super(ScalarReduce, self).__init__(runtime, task_id, lhs_dtype, tag)

    def execute(self, launch_domain, redop):
        argbuf = PandasBufferBuilder()
        return self._runtime.dispatch(
            self.build_task(launch_domain, argbuf), redop
        ).cast(self._lhs_dtype)

    def execute_single(self):
        argbuf = PandasBufferBuilder()
        return self._runtime.dispatch(self.build_single_task(argbuf)).cast(
            self._lhs_dtype
        )

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

from struct import unpack

import numpy as np

from legate.core import Point

from legate.pandas.common import types as ty
from legate.pandas.config import OpCode

from .pattern import ScalarMap


class Scalar(object):
    def __init__(self, runtime, dtype, valid, value):
        assert isinstance(valid, bool)
        self.dtype = dtype
        self.valid = valid

        self._runtime = runtime
        self._value = value

    @property
    def value(self):
        return (
            self._value
            if self.valid
            else (False if self.dtype == ty.bool else np.nan)
        )

    def astype(self, dtype):
        if self.dtype == dtype:
            return self
        else:
            return self._runtime.create_scalar(self._value, dtype)

    def get_future(self):
        return self._runtime.create_future_from_scalar(self)

    # the read_only parameter is unnecessary, but added to keep the interface
    # the same as the column's.
    def add_to_plan(self, plan, read_only=True):
        assert read_only
        plan.add_future(self.get_future())

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            f"Scalar(valid={self.valid},value={self._value},"
            f"dtype={self.dtype})"
        )


class PandasFuture(object):
    def __init__(self, runtime, future, dtype=None, ready=False):
        self.dtype = dtype
        self.ready = ready

        self._runtime = runtime
        self._future = future

    @property
    def handle(self):
        return self._future.handle

    def cast(self, dtype):
        return PandasFuture(self._runtime, self._future, dtype, self.ready)

    def equal_size(self, other):
        assert isinstance(other, PandasFuture)
        assert self.dtype is not None
        assert self.dtype == other.dtype

        if self.ready and other.ready:
            return self._runtime.create_scalar(
                self.get_value() == other.get_value(), ty.bool
            ).get_future()

        runtime = self._runtime
        plan = ScalarMap(runtime, OpCode.SIZES_EQUAL, ty.bool)
        plan.add_future(self._future)
        plan.add_future(other._future)
        return plan.execute_single()

    def get_value(self):
        if self.dtype is None:
            raise ValueError("Invalid get_value call to untyped future")
        return np.frombuffer(
            self._future.get_buffer(self.dtype.itemsize),
            dtype=self.dtype.to_pandas(),
            count=1,
        )[0]

    def get_scalar(self):
        buf = self._future.get_buffer()
        (
            valid,
            type_code,
        ) = unpack("ii", buf[:8])
        dtype = ty.code_to_dtype(type_code)
        assert not ty.is_string_dtype(dtype)
        assert dtype == self.dtype
        (value,) = unpack(
            ty.to_format_string(dtype), buf[8 : 8 + dtype.itemsize]
        )
        return Scalar(self._runtime, dtype, bool(valid), value)

    def wait(self):
        self._future.wait()

    def add_to_plan(self, plan, read_only=True):
        assert read_only
        plan.add_future(self)


class PandasFutureMap(object):
    def __init__(self, runtime, future_map, dtype=None):
        self.dtype = dtype

        self._runtime = runtime
        self._future_map = future_map

    @property
    def future_map(self):
        return self._future_map

    @property
    def handle(self):
        return self._future_map.handle

    def cast(self, dtype):
        return PandasFutureMap(self._runtime, self._future_map, dtype)

    def get_future(self, point):
        point = Point([point])
        return PandasFuture(
            self._runtime, self._future_map.get_future(point), self.dtype
        )

    def get_futures(self, num_points):
        return [self.get_future(idx) for idx in range(num_points)]

    def reduce(self, op, dtype=None, deterministic=False):
        if self.dtype is None:
            raise ValueError("Untyped future map cannot be reduced")
        if dtype is None:
            dtype = self.dtype
        return PandasFuture(
            self._runtime,
            self._runtime.reduce_future_map(
                self._future_map, op, dtype, deterministic
            ),
            dtype,
        )

    def sum(self):
        return self.reduce("sum")

    def wait(self):
        return self._future_map.wait()

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

from legate.pandas.common import types as ty
from legate.pandas.config import OpCode
from legate.pandas.core.pattern import Map, Projection


class Bitmask(object):
    alloc_type = ty.uint8

    def __init__(self, runtime, storage):
        self._runtime = runtime
        self._storage = storage
        self._compact_bitmask = None

    @property
    def region(self):
        return self._storage.region

    @property
    def field_id(self):
        return self._storage.field_id

    @property
    def dtype(self):
        return self._storage.dtype

    @property
    def num_pieces(self):
        return self._storage.num_pieces

    @property
    def launch_domain(self):
        return self._storage.launch_domain

    @property
    def storage(self):
        return self._storage.storage

    @property
    def ispace(self):
        return self._storage.ispace

    @property
    def primary_ipart(self):
        return self._storage.primary_ipart

    @property
    def fspace(self):
        return self._storage.fspace

    @property
    def compact_bitmask(self):
        if self._compact_bitmask is None:
            result = self._storage.storage.create_new_field(self.dtype)
            result.set_primary_ipart(self.primary_ipart)

            plan = Map(self._runtime, OpCode.TO_BITMASK)
            proj = Projection(self._storage.primary_ipart)

            plan.add_output_only(result)
            plan.add_scalar_arg(False, ty.bool)
            plan.add_scalar_arg(0, ty.uint32)

            plan.add_input(self._storage, proj)
            plan.add_scalar_arg(False, ty.bool)
            plan.add_scalar_arg(0, ty.uint32)

            plan.execute(self.launch_domain)

            self._compact_bitmask = result
        return self._compact_bitmask

    def clone(self):
        return Bitmask(self._runtime, self._storage.clone())

    def set_primary_ipart(self, ipart):
        self._storage.set_primary_ipart(ipart)

    def get_view(self, ipart):
        return self._runtime.create_region_partition(self.region, ipart)

    # XXX: This method should be used with caution as it gets blocked
    #      on a future internally
    def has_nulls(self):
        return self.count_nulls().sum().get_value() > 0

    def count_nulls(self):
        plan_count = Map(self._runtime, OpCode.COUNT_NULLS)
        proj = Projection(self._storage.primary_ipart)
        plan_count.add_input(self._storage, proj)
        return plan_count.execute(self.launch_domain).cast(ty.uint64)

    def to_raw_address(self):
        return self._storage.to_raw_address()

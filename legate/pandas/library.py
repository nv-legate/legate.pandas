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

import os

from legate.core import LegateLibrary, legate_add_library


class PandasLibrary(LegateLibrary):
    _LIBRARY_NAME = "legate.pandas"
    _MAX_NUM_TASKS = 1 << 20
    # We have only one reduction operator now
    _MAX_NUM_REDOPS = 1

    def __init__(self):
        # FIXME: Currently, we don't set up the library path correctly
        #        and sometimes fail to load all dependencies.
        #        Importing PyArrow makes sure those dependencies
        #        are loaded.
        import pyarrow  # noqa: F401

        self.pandas_lib = None
        legate_add_library(self)

    def get_name(self):
        return self._LIBRARY_NAME

    def get_shared_library(self):
        from legate.pandas.install_info import libpath

        return os.path.join(
            libpath, "liblgpandas" + self.get_library_extension()
        )

    def get_c_header(self):
        from legate.pandas.install_info import header

        return header

    def get_registration_callback(self):
        return "legate_pandas_perform_registration"

    def get_cuda_arch(self):
        assert self.pandas_lib is not None
        cuda_arch = int(self.pandas_lib.legate_pandas_get_cuda_arch())
        if cuda_arch == 0xFFFFFFFF:
            import numba.cuda

            device = numba.cuda.get_current_device()
            return device.compute_capability
        else:
            return (cuda_arch / 10, cuda_arch % 10)

    def get_use_nccl(self):
        assert self.pandas_lib is not None
        return bool(self.pandas_lib.legate_pandas_use_nccl())

    def set_runtime(self, runtime):
        self._runtime = runtime

    def initialize(self, pandas_lib):
        assert self.pandas_lib is None
        self.pandas_lib = pandas_lib

    def destroy(self):
        assert self._runtime is not None
        self._runtime.destroy()

    def generate_ids(self, runtime):
        from legate.core import legion

        from legate.pandas.config import ProjectionCode

        name = self._LIBRARY_NAME.encode("utf-8")

        return (
            legion.legion_runtime_generate_library_task_ids(
                runtime, name, self._MAX_NUM_TASKS
            ),
            legion.legion_runtime_generate_library_projection_ids(
                runtime,
                name,
                ProjectionCode.LAST_PROJ,
            ),
            legion.legion_runtime_generate_library_reduction_ids(
                runtime, name, self._MAX_NUM_REDOPS
            ),
            legion.legion_runtime_generate_library_mapper_ids(
                runtime, name, 1
            ),
        )


library = PandasLibrary()
c_header = library.pandas_lib

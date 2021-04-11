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


def _invalid_value_error(name, value):
    raise ValueError(f"invalid '{name}': {value}")


def _unsupported_error(msg, value=None):
    if value is not None:
        return NotImplementedError(f"'{msg}'={value} is not supported yet")
    else:
        return NotImplementedError(f"{msg}")


def _warning(msg):
    warnings.warn(msg)

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


from legate.pandas.common import errors as err
from legate.pandas.core.runtime import _runtime as rt


def _check_legate_data(legate_data):
    if legate_data["version"] != 1:
        raise err._unsupported_error(
            "Unsupported legate data interface version: "
            f"{legate_data['version']}"
        )

    if "data" not in legate_data:
        raise ValueError("Invalid Legate Data Interface object")


def from_legate_data(legate_data):
    _check_legate_data(legate_data)

    return rt.create_dataframe_from_legate_data(legate_data["data"])


def from_named_legate_data(named_legate_data):
    columns = list(named_legate_data.keys())

    all_legate_data = dict()
    for key, item in named_legate_data.items():
        legate_data = item.__legate_data_interface__
        _check_legate_data(legate_data)

        for field, array in legate_data["data"].items():
            field = field.with_name(key)
            all_legate_data[field] = array

    return rt.create_dataframe_from_legate_data(all_legate_data), columns


def from_pandas(df, index=None):
    return rt.create_dataframe_from_pandas(df, index=index)

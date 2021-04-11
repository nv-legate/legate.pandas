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

import legate.numpy as np
from legate import pandas as lp

s1 = lp.Series([1, 2, 3])
data1 = s1.__legate_data_interface__
assert data1["version"] == 1
assert len(data1["data"]) == 1
for field, array in data1["data"].items():
    assert field.name == "column0"
    assert str(field.type) == "int64"
    assert not field.nullable
    stores = array.stores()
    assert len(stores) == 2
    assert stores[0] is None

arr1 = np.array(s1)
arr2 = np.array([1, 2, 3])
assert np.array_equal(arr1, arr2)

s2 = lp.Series([1.0, np.nan, 3.0])
data2 = s2.__legate_data_interface__
assert len(data2["data"]) == 1
for field, array in data2["data"].items():
    assert field.name == "column0"
    assert str(field.type) == "double"
    assert field.nullable
    stores = array.stores()
    assert len(stores) == 2
    assert stores[0] is not None

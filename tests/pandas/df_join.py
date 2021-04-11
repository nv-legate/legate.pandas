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

import itertools

import numpy as np
import pandas as pd
from numpy.random import randn

from legate import pandas as lp
from tests.utils import equals

for n in [15, 30, 45]:
    c1 = np.array(randn(n) * 100.0, dtype="int64")
    c2 = np.array(randn(n // 3) * 100.0, dtype="int64")
    c3_l = np.array(randn(n) * 100.0, dtype="int64")
    c3_r = np.array(randn(n // 3) * 100.0, dtype="int64")

    key_left = np.array(
        list(itertools.chain(*[[x] * 3 for x in range(n // 3 - 1, -1, -1)])),
        dtype="int64",
    )
    key_right = np.array(list(range(n // 3)), dtype="int64")

    print(f"Type: left, Size: {n}")

    df1 = pd.DataFrame(
        {"c1": c1, "key1": key_left, "key2": key_left, "c3": c3_l}
    )
    df2 = pd.DataFrame(
        {"c2": c2, "key1": key_right, "key2": key_right, "c3": c3_r}
    )

    ldf1 = lp.DataFrame(df1)
    ldf2 = lp.DataFrame(df2)

    join_pandas = df1.join(
        df2.set_index(["key1", "key2"]), on=["key1", "key2"], lsuffix="_l"
    )
    join_legate = ldf1.join(
        ldf2.set_index(["key1", "key2"]), on=["key1", "key2"], lsuffix="_l"
    )

    join_pandas = join_pandas.sort_index()
    join_legate = join_legate.sort_index()

    assert equals(join_legate, join_pandas)

    join_pandas = df1.join(df2, on="key1", lsuffix="_l")
    join_legate = ldf1.join(ldf2, on="key1", lsuffix="_l")

    join_pandas = join_pandas.sort_index()
    join_legate = join_legate.sort_index()

    assert equals(join_legate, join_pandas)

    df1 = df1.set_index(["key2", "key1"])
    df2 = df2.set_index(["key2", "key1"])

    ldf1 = ldf1.set_index(["key2", "key1"])
    ldf2 = ldf2.set_index(["key2", "key1"])

    join_pandas = df1.join(df2, lsuffix="_l")
    join_legate = ldf1.join(ldf2, lsuffix="_l")

    join_pandas = join_pandas.sort_index()
    join_legate = join_legate.sort_index()

    assert equals(join_legate, join_pandas)

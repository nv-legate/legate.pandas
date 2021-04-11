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

key_dtypes = [np.int32, np.int64, np.float32, np.float64]


def to_pandas(ldf):
    columns = {}
    for column in ldf.columns:
        columns[column] = ldf[column].to_numpy()
    return pd.DataFrame(columns)


def sort_and_compare(df1, df2):
    df1 = df1.sort_values(by=df1.columns.to_list(), ignore_index=True)
    df2 = df2.sort_values(by=df1.columns.to_list(), ignore_index=True)
    return df1.equals(df2)


for n in [15, 30, 45]:
    c1 = np.array(randn(n) * 100.0, dtype=np.int64)
    c2 = np.array(randn(n // 3) * 100.0, dtype=np.int64)
    c3 = np.array(randn(n // 3) * 100.0, dtype=np.int64)
    c4 = np.array(randn(n // 3) * 100.0, dtype=np.int64)

    key1 = list(itertools.chain(*[[x] * 3 for x in range(n // 3 - 1, -1, -1)]))
    key2 = list(range(n // 3))
    key3 = list(range(n // 3))

    df1 = pd.DataFrame(
        {
            "c1": c1,
            "key1": np.array(key1, dtype=np.int64),
            "key2": np.array(key1[::-1], dtype=np.int64),
        }
    )
    df2 = pd.DataFrame(
        {
            "c2": c2,
            "key1": np.array(key2, dtype=np.int64),
            "key2": np.array(key2[::-1], dtype=np.int64),
        }
    )
    df3 = pd.DataFrame({"c3": c3, "key1": np.array(key3, dtype=np.int64)})
    df4 = pd.DataFrame(
        {
            "c4": c3,
            "key1": np.array(key3, dtype=np.int64),
            "key2": np.array(key3[::-1], dtype=np.int64),
        }
    )

    ldf1 = lp.DataFrame(df1)
    ldf2 = lp.DataFrame(df2)
    ldf3 = lp.DataFrame(df3)
    ldf4 = lp.DataFrame(df4)

    join_pandas = (
        df1.merge(df2, on=["key1", "key2"])
        .merge(df3, on="key1")
        .merge(df4, on="key1")
    )
    join_legate = (
        ldf1.merge(ldf2, on=["key1", "key2"])
        .merge(ldf3, on="key1")
        .merge(ldf4, on="key1")
    )

    assert sort_and_compare(join_pandas, to_pandas(join_legate))

    join_pandas = (
        df1.merge(df2, on=["key1", "key2"])
        .merge(df3, on="key1")
        .merge(df4, on=["key1", "key2"])
    )
    join_legate = (
        ldf1.merge(ldf2, on=["key1", "key2"])
        .merge(ldf3, on="key1")
        .merge(ldf4, on=["key1", "key2"])
    )

    assert sort_and_compare(join_pandas, to_pandas(join_legate))

    join_pandas = df1.merge(df2, on=["key1", "key2"]).merge(
        df4, on=["key1", "key2"]
    )
    join_legate = ldf1.merge(ldf2, on=["key1", "key2"]).merge(
        ldf4, on=["key1", "key2"]
    )

    assert sort_and_compare(join_pandas, to_pandas(join_legate))

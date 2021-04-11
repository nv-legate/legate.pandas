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


def to_pandas(ldf):
    columns = {}
    for column in ldf.columns:
        columns[column] = ldf[column].to_numpy()
    return pd.DataFrame(columns)


def sort_and_compare(df1, df2):
    df1 = df1.sort_values(by=df1.columns.to_list(), ignore_index=True)
    df2 = df2.sort_values(by=df1.columns.to_list(), ignore_index=True)
    return df1.equals(df2)


key_dtype = np.float64

for n in [15, 30, 45]:
    c1 = np.array(randn(n) * 100.0, dtype=np.float64)
    c2 = np.array(randn(n // 3) * 100.0, dtype=np.float64)
    c3_l = np.array(randn(n) * 100.0, dtype=np.float64)
    c3_r = np.array(randn(n // 3) * 100.0, dtype=np.float64)

    for i in range(len(c1)):
        if i % 5 == 0:
            c1[i] = np.nan
    for i in range(len(c2)):
        if i % 4 == 0:
            c2[i] = np.nan

    key_right = list(range(n // 3))
    key_left = list(
        itertools.chain(*[[x] * 3 for x in range(n // 3 - 1, -1, -1)])
    )
    for i in range(len(key_left)):
        if i % 4 == 3:
            key_left[i] = np.nan
    for i in range(len(key_right)):
        if i % 3 == 0:
            key_right[i] = np.nan

    print("Type: inner, Size: %u, Key dtype: %s" % (n, str(key_dtype)))

    df1 = pd.DataFrame(
        {"c1": c1, "key": np.array(key_left, dtype=key_dtype), "c3": c3_l}
    )
    df2 = pd.DataFrame(
        {"c2": c2, "key": np.array(key_right, dtype=key_dtype), "c3": c3_r}
    )

    ldf1 = lp.DataFrame(df1)
    ldf2 = lp.DataFrame(df2)

    join_pandas = df1.merge(df2, on="key").fillna(9999.0)
    join_legate = ldf1.merge(ldf2, on="key", method="broadcast").fillna(9999.0)
    join_legate_hash = ldf1.merge(ldf2, on="key", method="hash").fillna(9999.0)

    assert sort_and_compare(join_pandas, to_pandas(join_legate))
    assert sort_and_compare(join_pandas, to_pandas(join_legate_hash))

    key_left = list(itertools.chain(*[[x] * 3 for x in range(n // 3, 0, -1)]))
    for i in range(len(key_left)):
        if i % 4 == 0:
            key_left[i] = np.nan

    print("Type: left, Size: %u, Key dtype: %s" % (n, str(key_dtype)))

    df1 = pd.DataFrame(
        {"c1": c1, "key": np.array(key_left, dtype=key_dtype), "c3": c3_l}
    )
    df2 = pd.DataFrame(
        {"c2": c2, "key": np.array(key_right, dtype=key_dtype), "c3": c3_r}
    )

    ldf1 = lp.DataFrame(df1)
    ldf2 = lp.DataFrame(df2)

    join_pandas = df1.merge(df2, on="key", how="left").fillna(9999.0)
    join_legate = ldf1.merge(
        ldf2, on="key", how="left", method="broadcast"
    ).fillna(9999.0)
    join_legate_hash = ldf1.merge(
        ldf2, on="key", how="left", method="hash"
    ).fillna(9999.0)

    assert sort_and_compare(join_pandas, to_pandas(join_legate))
    assert sort_and_compare(join_pandas, to_pandas(join_legate_hash))

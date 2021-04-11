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


for n in [15, 30, 45]:
    categories1 = ["C" + str(i) for i in range(-1, n // 3)]
    categories1.sort()
    key_dtype1 = pd.CategoricalDtype(categories=categories1)

    categories2 = ["C" + str(i) for i in range(0, n // 3 * 2, 2)]
    categories2.sort()
    key_dtype2 = pd.CategoricalDtype(categories=categories2)

    c1 = np.array(randn(n) * 100.0, dtype=np.float64)
    c2 = np.array(randn(n // 3) * 100.0, dtype=np.float64)
    c3_l = np.array(randn(n) * 100.0, dtype=np.float64)
    c3_r = np.array(randn(n // 3) * 100.0, dtype=np.float64)

    key_left = pd.Categorical.from_codes(
        list(itertools.chain(*[[x] * 3 for x in range(n // 3 - 2, -2, -1)])),
        dtype=key_dtype1,
    )
    key_right = pd.Categorical.from_codes(
        list(range(-1, len(categories2) - 1)), dtype=key_dtype2
    )
    print("Type: inner, Size: %u" % n)
    print("Categories 1: %s" % str(list(key_dtype1.categories)))
    print("Categories 2: %s" % str(list(key_dtype2.categories)))

    df1 = pd.DataFrame({"c1": c1, "key": key_left, "c3": c3_l})
    df2 = pd.DataFrame({"c2": c2, "key": key_right, "c3": c3_r})

    ldf1 = lp.DataFrame(df1)
    ldf2 = lp.DataFrame(df2)

    join_pandas = df1.merge(df2, on="key")
    join_pandas["key"] = join_pandas["key"].astype(object)
    join_legate = ldf1.merge(ldf2, on="key")

    assert sort_and_compare(join_pandas, to_pandas(join_legate))

    join_pandas = df1.merge(df2, on="key", how="left")
    join_pandas["key"] = join_pandas["key"].astype(object)
    join_legate = ldf1.merge(ldf2, on="key", how="left")

    assert sort_and_compare(join_pandas, to_pandas(join_legate))

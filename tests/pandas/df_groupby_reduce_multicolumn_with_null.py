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

from itertools import product

import numpy as np
import pandas as pd
from numpy.random import randn
from pandas.core.groupby import DataFrameGroupBy as GroupBy

from legate import pandas as lp
from legate.pandas.frontend.groupby import GroupBy as LegateGroupBy

aggs = ["sum", "min", "max"]
n = 32


def to_pandas(ldf):
    columns = {}
    for column in ldf.columns:
        columns[column] = ldf[column].to_numpy()
    return pd.DataFrame(columns, index=ldf.index)


def sort_and_compare(df1, df2):
    df1 = df1.sort_index()
    df2 = df2.sort_index()
    return df1.equals(df2)


for pair in product(aggs, aggs[1:] + aggs[:1]):
    agg1, agg2 = pair
    if agg1 == agg2:
        f = getattr(GroupBy, agg1)
        mf = getattr(LegateGroupBy, agg1)
    else:

        def f(g):
            return getattr(GroupBy, "agg")(g, {"c1": agg1, "c2": agg2})

        def mf(g):
            return getattr(LegateGroupBy, "agg")(g, {"c1": agg1, "c2": agg2})

    keys1 = [1, 4, 2, 3, 1, 3, 1]
    keys2 = [1.0, 4.0, np.nan, 3, np.nan, 2, 1]
    key_dtype1 = np.int64
    key_dtype2 = np.float64
    print(
        "Agg for c1: %s, Agg for c2: %s, Key type1: %s, Key type2: %s"
        % (agg1, agg2, str(key_dtype1), str(key_dtype2))
    )
    df = pd.DataFrame(
        {
            "c1": np.array(randn(n) * 100.0, dtype=np.int64),
            "c2": np.array([np.nan] * n, dtype=np.float64),
            "c3": np.array(
                (keys1 * ((n + len(keys1) - 1) // len(keys1)))[:n],
                dtype=np.dtype(key_dtype1),
            ),
            "c4": np.array(
                (keys2 * ((n + len(keys2) - 1) // len(keys2)))[:n],
                dtype=np.dtype(key_dtype2),
            ),
        }
    )
    ldf = lp.DataFrame(df)

    out_df = f(df.groupby(["c3", "c4"], sort=False))
    out_df.index.names = ["c5", "c6"]
    out_ldf = mf(ldf.groupby(["c3", "c4"], sort=False, method="hash"))
    # XXX: Note that sum reductions in Pandas yield 0.0 when all values are
    #      NaNs, whereas both Legate and cuDF produce nulls in that case.
    #      Here we replace NaNs/nulls with 0.0 to ignore this semantics
    #      difference.
    out_df = out_df.fillna(0.0)
    out_ldf = out_ldf.fillna(0.0)
    assert sort_and_compare(out_df, to_pandas(out_ldf))

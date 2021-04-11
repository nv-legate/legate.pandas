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
from tests.utils import equals

aggs = ["sum", "var", "std", "mean", "count", "size"]
n = 32


def similar_series(a, b):
    return ((a - b).abs() < 1e-10).all()


def similar(a, b):
    if isinstance(a, pd.DataFrame):
        for c in a.columns:
            if not similar_series(a[c], b[c]):
                return False
        return True
    else:
        return similar_series(a, b)


def to_pandas(ldf):
    if isinstance(ldf, lp.DataFrame):
        columns = {}
        for column in ldf.columns:
            columns[column] = ldf[column].to_numpy()
        return pd.DataFrame(columns, index=ldf.index)
    else:
        return pd.Series(ldf.to_numpy(), index=ldf.index)


def sort_and_compare(df1, df2):
    df1 = df1.sort_index()
    df2 = df2.sort_index()
    return df1.equals(df2) or similar(df1, df2)


keys = [1, 4, 2, np.nan, 3, 1, 3, 1, np.nan]
c1 = np.array(randn(n) * 100.0, dtype=np.float64)
c2 = np.array(randn(n) * 200.0, dtype=np.float64)

for i in range(n):
    if i % 3 == 0:
        c1[i] = np.nan

df = pd.DataFrame(
    {
        "c1": c1,
        "c2": c2,
        "c3": np.array(
            (keys * ((n + len(keys) - 1) // len(keys)))[:n], dtype="float64"
        ),
        "c4": np.array(randn(n) * 200.0, dtype=np.float64),
    }
)
ldf = lp.DataFrame(df)

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

    print(f"Agg for c1: {agg1}, Agg for c2: {agg2}")
    out_pd = f(df.groupby("c3", sort=True))
    out_lp = mf(ldf.groupby("c3", method="hash", sort=True))
    assert equals(out_lp, out_pd)

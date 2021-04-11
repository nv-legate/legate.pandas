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

aggs = ["sum", "count", "mean", "var", "std"]
n = 32

keys = [1, 4, 2, 3, 1, 3, 1]
df = pd.DataFrame(
    {
        "c1": np.array(randn(n) * 100.0, dtype=np.int64),
        "c2": np.array(randn(n) * 200.0, dtype=np.int64),
        "c3": np.array(
            (keys * ((n + len(keys) - 1) // len(keys)))[:n],
            dtype=np.dtype(np.int64),
        ),
        "c4": np.array(randn(n) * 200.0, dtype=np.int64),
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

    out_df = f(df.groupby("c3", sort=True))
    out_ldf = mf(ldf.groupby("c3", sort=True))
    for col, dtype in out_df.dtypes.items():
        out_ldf[col] = out_ldf[col].astype(dtype)
    assert equals(out_ldf, out_df)

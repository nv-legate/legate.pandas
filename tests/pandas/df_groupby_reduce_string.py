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

aggs = ["sum", "min", "max"]
n = 30

key_dtype = pd.StringDtype()
keys = ["C0", "C3", "C1", "C2", "C0", "C2", "C0"]
c1 = np.array(randn(n) * 100.0, dtype=np.int64)
c2 = np.array(randn(n) * 200.0, dtype=np.int64)
c3 = (keys * ((n + len(keys) - 1) // len(keys)))[:n]

df = pd.DataFrame({"c1": c1, "c2": c2, "c3": c3})
df["c3"] = df["c3"].astype(key_dtype)
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

    print(
        "Agg for c1: %s, Agg for c2: %s, Key type: %s"
        % (agg1, agg2, str(key_dtype))
    )
    out_df = f(df.groupby("c3", sort=False))
    # FIXME: Sorting on string columns is broken.
    #        For now, we turn off the sorting for groupby and
    #        convert the result to pandas before the comparison.
    out_ldf = mf(ldf.groupby("c3", sort=False, method="hash"))

    out_df = out_df.sort_index()
    out_ldf = out_ldf.to_pandas().sort_index()
    assert equals(out_ldf, out_df)

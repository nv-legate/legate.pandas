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

import numpy as np
import pandas as pd
from numpy.random import permutation

from legate import pandas as lp

n = 17

a = list(range(n))
b = [str(i) * 3 for i in range(n)]
c = [str(i % 3) for i in range(n)]

for i in range(n):
    if i % 3 == 0:
        a[i] = np.nan
    if i % 4 == 0:
        b[i] = None
    if i % 5 == 0:
        c[i] = None

for index in [
    pd.RangeIndex(n),
    pd.RangeIndex(1, n * 2 + 1, 2),
    pd.Index(permutation(n)),
]:
    df = pd.DataFrame({"a": a, "b": b, "c": c}, index=index)
    df["b"] = df["b"].astype(pd.StringDtype())
    df["c"] = df["c"].astype("category")

    ldf = lp.DataFrame(df)

    for how in ["any", "all"]:
        out_pd = df.dropna(how=how)
        out_lp = ldf.dropna(how=how)
        assert out_lp.equals(out_pd)

    for threshold in range(3):
        out_pd = df.dropna(thresh=threshold)
        out_lp = ldf.dropna(thresh=threshold)
        assert out_lp.equals(out_pd)

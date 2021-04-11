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

from legate import pandas as lp
from tests.utils import equals

n = 17
a = list(range(n))
b = list(range(1, n + 1))
c = [str(i) * 3 for i in range(n)]
d = [i % 3 for i in range(n)]
e = [np.datetime64(f"2021-03-{i + 10}") for i in range(n)]

for i in range(n):
    if i % 4 == 0:
        a[i] = np.nan
    if i % 3 == 0:
        c[i] = None
    if i % 5 == 0:
        b[i] = np.nan
        d[i] = -1
        e[i] = pd.NaT

df = pd.DataFrame(
    {
        "a": a,
        "b": b,
        "c": pd.Series(c, dtype=pd.StringDtype()),
        "d": pd.Categorical.from_codes(d, ["cat1", "cat2", "cat3"]),
        "e": e,
    }
)
ldf = lp.DataFrame(df)

assert equals(ldf[["a", "b"]].fillna(1), df[["a", "b"]].fillna(1))
assert equals(
    ldf[["e"]].fillna(lp.to_datetime("1984-04-06")),
    df[["e"]].fillna(pd.to_datetime("1984-04-06")),
)
assert equals(
    ldf[["e"]].fillna(pd.to_datetime("1984-04-06")),
    df[["e"]].fillna(pd.to_datetime("1984-04-06")),
)
assert equals(
    ldf.fillna({"a": 1, "c": "ABC", "d": "cat1"}),
    df.fillna({"a": 1, "c": "ABC", "d": "cat1"}),
)
assert equals(
    ldf[["b", "e"]].fillna({"a": 1, "c": "ABC", "d": "cat1"}),
    df[["b", "e"]].fillna({"a": 1, "c": "ABC", "d": "cat1"}),
)

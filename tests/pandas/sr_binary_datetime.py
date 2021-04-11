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

start = pd.to_datetime("2015-01-01")

n = 30
s1 = pd.Series(
    [start + pd.Timedelta(f"{idx}d") for idx in np.random.randint(0, 10, n)],
    dtype=np.dtype("datetime64[ns]"),
)
s2 = pd.Series(
    [start + pd.Timedelta(f"{idx}d") for idx in np.random.randint(0, 10, n)],
    dtype=np.dtype("datetime64[ns]"),
)

for i in range(n):
    if i % 3 == 0:
        s1[i] = pd.NaT
    if i % 4 == 0:
        s2[i] = pd.NaT

ls1 = lp.Series(s1)
ls2 = lp.Series(s2)

ops = [
    "lt",
    "gt",
    "le",
    "ge",
    "ne",
]

print("##### Testing normal operators #####")
for op in ops:
    print("Testing " + op)
    f = getattr(pd.Series, op)
    out_s = f(s1, s2)
    f = getattr(lp.Series, op)
    out_ls = f(ls1, ls2)
    assert equals(out_ls, out_s)

scalar = start + pd.Timedelta("5d")

print("##### Testing broadcast operators #####")
for op in ops:
    print("Testing " + op)
    f = getattr(pd.Series, op)
    out_s = f(s1, scalar)
    f = getattr(lp.Series, op)
    out_ls = f(ls1, scalar)
    assert equals(out_ls, out_s)

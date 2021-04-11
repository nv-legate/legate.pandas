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

import pandas as pd
from numpy.random import randint, randn

from legate import pandas as lp

df1 = pd.DataFrame({1: randn(10), 2: randn(10), 5: randn(10)})
ldf1 = lp.DataFrame(df1)
df2 = pd.DataFrame({1: randn(10), 2: randn(10), 5: randn(10)})
ldf2 = lp.DataFrame(df2)

ops = [
    "add",
    "sub",
    "mul",
    "div",
    "truediv",
    "floordiv",
    "mod",
    # TODO: nans_to_nulls is required to match the pandas result
    # "pow",
    "radd",
    "rsub",
    "rmul",
    "rdiv",
    "rtruediv",
    "rfloordiv",
    "mod",
]


def similar(a, b):
    for c in a.columns:
        if not ((a[c] - b[c]).abs() < 1e-14).all():
            return False
    return True


for op in ops:
    print("Testing " + op)
    f = getattr(pd.DataFrame, op)
    out_pd = f(df1, df2).fillna(0.0)
    f = getattr(lp.DataFrame, op)
    out_lp = f(ldf1, ldf2).fillna(0.0)
    assert out_lp.equals(out_pd) or similar(out_lp, out_pd)

ops = [
    "lt",
    "gt",
    "le",
    "ge",
    "ne",
]
for op in ops:
    print("Testing " + op)
    f = getattr(pd.DataFrame, op)
    out_pd = f(df1, df2)
    f = getattr(lp.DataFrame, op)
    out_lp = f(ldf1, ldf2)
    assert out_lp.equals(out_pd)

df1 = pd.DataFrame(
    {1: randint(0, 100, 10), 2: randint(0, 100, 10), 5: randint(0, 100, 10)}
)
ldf1 = lp.DataFrame(df1)
df2 = pd.DataFrame(
    {1: randint(0, 100, 10), 2: randint(0, 100, 10), 5: randint(0, 100, 10)}
)
ldf2 = lp.DataFrame(df2)

ops = ["__or__", "__and__", "__xor__"]

for op in ops:
    print("Testing " + op)
    f = getattr(pd.DataFrame, op)
    out_pd = f(df1, df2).fillna(0.0)
    f = getattr(lp.DataFrame, op)
    out_lp = f(ldf1, ldf2).fillna(0.0)
    assert out_lp.equals(out_pd)

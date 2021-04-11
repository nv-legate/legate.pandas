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
from numpy import nan
from numpy.random import randn

from legate import pandas as lp


def similar(a, b):
    return ((a - b).abs() < 1e-14).all()


for n in [100]:
    s1 = pd.Series(randn(1, n)[0])
    s2 = pd.Series(randn(1, n)[0])

    for i in range(n):
        if (i + 1) % 4 == 0:
            s1[i] = nan
        if (i + 1) % 3 == 0:
            s2[i] = nan

    ls1 = lp.Series(s1)
    ls2 = lp.Series(s2)

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
    ]

    for op in ops:
        print("Testing " + op)
        f = getattr(pd.Series, op)
        out_s = f(s1, s2)
        f = getattr(lp.Series, op)
        out_ls = f(ls1, ls2)
        out_s = out_s.fillna(9999.0)
        out_ls = out_ls.fillna(9999.0)
        assert out_ls.equals(out_s) or similar(out_ls, out_s)

    ops = [
        "lt",
        "gt",
        "le",
        "ge",
        "ne",
    ]

    for op in ops:
        print("Testing " + op)
        f = getattr(pd.Series, op)
        out_s = f(s1, s2)
        f = getattr(lp.Series, op)
        out_ls = f(ls1, ls2)
        assert out_ls.equals(out_s)

# Test for a corner case

a1 = [
    0.532274,
    -0.530682,
    -0.354287,
    nan,
    -0.604258,
    -0.224359,
    -0.504423,
    nan,
    -0.181599,
    0.427099,
]
a2 = [
    0.228509,
    1.429084,
    nan,
    0.275985,
    -1.644658,
    nan,
    -0.667356,
    -0.073481,
    nan,
    1.214095,
]

for k in range(100):
    s1 = pd.Series(a1)
    s2 = pd.Series(a2)

    ls1 = lp.Series(s1)
    ls2 = lp.Series(s2)

    out_s = s1.mod(s2)
    out_ls = ls1.mod(ls2)
    out_s = out_s.fillna(9999.0)
    out_ls = out_ls.fillna(9999.0)
    assert out_ls.equals(out_s) or similar(out_ls, out_s)

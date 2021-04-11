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

s1 = pd.Series(randn(1, 10)[0])
ls1 = lp.Series(s1)
s2 = pd.Series(randn(1, 10)[0])
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
    "lt",
    "gt",
    "le",
    "ge",
    "ne",
]


def similar(a, b):
    return ((a - b).abs() < 1e-14).all()


for op in ops:
    print("Testing " + op)
    f = getattr(pd.Series, op)
    out_s = f(s1, s2).fillna(0.0)
    f = getattr(lp.Series, op)
    out_ls = f(ls1, ls2).fillna(0.0)
    assert out_ls.equals(out_s) or similar(out_ls, out_s)

s1 = pd.Series(randint(0, 100, 10))
ls1 = lp.Series(s1)
s2 = pd.Series(randint(0, 100, 10))
ls2 = lp.Series(s2)

ops = ["__or__", "__and__", "__xor__"]

for op in ops:
    print("Testing " + op)
    f = getattr(pd.Series, op)
    out_s = f(s1, s2).fillna(0.0)
    f = getattr(lp.Series, op)
    out_ls = f(ls1, ls2).fillna(0.0)
    assert out_ls.equals(out_s)

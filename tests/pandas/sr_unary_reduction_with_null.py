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

import math

import pandas as pd
from numpy import nan
from numpy.random import randn

from legate import pandas as lp

ops = ["sum", "prod", "count", "min", "max", "var", "std", "mean"]

print("##### Testing normal inputs #####")

for n in [10, 100]:
    s = pd.Series(randn(1, n)[0])
    for i in range(n):
        if (i + 1) % 4 == 0:
            s[i] = nan
    ls = lp.Series(s)

    for op in ops:
        print("Testing " + op)
        f = getattr(pd.Series, op)
        out_pd = f(s)
        f = getattr(lp.Series, op)
        out_lp = f(ls)

        assert math.fabs(out_pd - out_lp) < 1e-14


s = pd.Series(randn(1, 10)[0])
for i in range(10):
    s[i] = nan
ls = lp.Series(s)


print("##### Testing all-null case #####")

for op in ops[3:]:
    print("Testing " + op)
    f = getattr(pd.Series, op)
    out_pd = f(s)
    f = getattr(lp.Series, op)
    out_lp = f(ls)

    assert math.isnan(out_pd)
    assert math.isnan(out_lp)

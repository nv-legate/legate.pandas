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
from numpy.random import randint

from legate import pandas as lp

n = 23

a1 = ["a" * i for i in randint(0, 5, n)]
a2 = ["a" * i for i in randint(0, 5, n)]

s1 = pd.Series(a1, dtype=pd.StringDtype())
s2 = pd.Series(a2, dtype=pd.StringDtype())

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
    # TODO: We cast the output, which has nullable boolean type,
    #       to a non-nullable booelean type. This casting will
    #       become unnecessary once #67 is addressed.
    out_s = f(s1, s2).astype("bool")
    f = getattr(lp.Series, op)
    out_ls = f(ls1, ls2).astype("bool")
    assert out_ls.equals(out_s)

print("##### Testing broadcast operators #####")
for op in ops:
    print("Testing " + op)
    f = getattr(pd.Series, op)
    # TODO: We cast the output, which has nullable boolean type,
    #       to a non-nullable booelean type. This casting will
    #       become unnecessary once #67 is addressed.
    out_s = f(s1, "a").astype("bool")
    f = getattr(lp.Series, op)
    out_ls = f(ls1, "a").astype("bool")
    assert out_ls.equals(out_s)

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

n = 14

strs = [
    "".join(
        [
            chr(ord("A") + c) if c < 3 else chr(ord("a") + (c - 3))
            for c in randint(0, 5, 1)
        ]
    )
    for _ in range(n)
]
s = pd.Series(strs, dtype=pd.StringDtype())
ls = lp.Series(s)

cat_s = s.astype("category")
cat_ls = ls.astype("category")
assert cat_ls.equals(cat_s)

cat = pd.CategoricalDtype(["b", "c", "B", "C"])
cat_s = cat_s.astype(cat)
cat_ls = cat_ls.astype(cat)
assert cat_ls.equals(cat_s)

cat_s = s.astype(cat)
cat_ls = ls.astype(cat)
assert cat_ls.equals(cat_s)

# The following should be no-ops
cat_s = cat_s.astype("category")
cat_ls = cat_ls.astype("category")
assert cat_ls.equals(cat_s)

str_s = cat_s.astype("string")
str_ls = cat_ls.astype("string")
assert str_ls.equals(str_s)

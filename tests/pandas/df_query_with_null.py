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

a = randn(10)
b = randn(10)
c = randn(10)

for i in range(10):
    if i % 3 == 0:
        a[i] = nan

for i in range(10):
    if i % 4 == 0:
        b[i] = nan

df = pd.DataFrame({"a": a, "b": b, "c": c})
ldf = lp.DataFrame(df)

delta = 0.001
query = "(a + b) / 2.0 > c + @delta"

print("Query: " + query)
out_pd = df.query(query)
out_lp = ldf.query(query)
assert out_lp.equals(out_pd)

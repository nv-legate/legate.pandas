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
from numpy.random import randn

from legate import pandas as lp
from tests.utils import equals

df1 = pd.DataFrame(
    {
        1: randn(10),
        2: randn(10),
        5: randn(10),
        6: randn(10),
    }
)
df1 = df1.rename(columns={2: 1})
ldf1 = lp.DataFrame(df1)
df2 = pd.DataFrame({1: randn(10), 3: randn(10), 5: randn(10)})
df2 = df2.rename(columns={3: 5})
ldf2 = lp.DataFrame(df2)

ops = ["add", "sub", "mul", "div", "truediv", "floordiv"]

for op in ops:
    print("Testing " + op)

    f = getattr(pd.DataFrame, op)
    out_pd = f(df1, df2, fill_value=1.0)
    f = getattr(lp.DataFrame, op)
    out_lp = f(ldf1, ldf2, fill_value=1.0)
    assert equals(out_lp, out_pd)

    f = getattr(pd.DataFrame, op)
    out_pd = f(df1, df2, fill_value=1.0)
    f = getattr(lp.DataFrame, op)
    out_lp = f(ldf1, ldf2, fill_value=1.0)
    assert equals(out_lp, out_pd)

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

df = pd.DataFrame({"c1": randn(1, 10)[0], "c2": randn(1, 10)[0]})
df.iloc[range(10)[::2], 0] = nan
df.iloc[range(10)[1::2], 1] = nan
ldf = lp.DataFrame(df)

ops = ["abs", "isna", "notna"]

for op in ops:
    print("Testing " + op)
    f = getattr(pd.DataFrame, op)
    out_df = f(df).fillna(1.0)
    f = getattr(lp.DataFrame, op)
    out_ldf = f(ldf).fillna(1.0)
    assert out_ldf.equals(out_df)

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
from numpy.random import permutation, randn

from legate import pandas as lp
from tests.utils import equals

n = 17
indices = [pd.RangeIndex(1, n + 1), pd.Index(permutation(n))]

for index in indices:
    print(f"Index: {index}")
    df1 = pd.DataFrame({1: randn(n), 2: randn(n), 5: randn(n)}, index=index)
    ldf1 = lp.DataFrame(df1)
    df2 = pd.DataFrame({1: randn(n), 2: randn(n), 5: randn(n)}, index=index)

    out_pd = df1 + df2
    out_lp = ldf1 + df2
    assert equals(out_lp, out_pd)

    out_pd = df1 + df2.values
    out_lp = ldf1 + df2.values
    assert equals(out_lp, out_pd)

    out_pd = df1.add(df2[1].values, axis=0)
    out_lp = ldf1.add(df2[1].values, axis=0)
    assert equals(out_lp, out_pd)

    out_pd = df1.add(df2[1].to_list(), axis=0)
    out_lp = ldf1.add(df2[1].to_list(), axis=0)
    assert equals(out_lp, out_pd)

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
from numpy.random import randint, randn

from legate import pandas as lp

ops = ["cumsum", "cumprod", "cummin", "cummax"]


def similar(a, b):
    if not ((a - b).abs() < 1e-14).all():
        return False
    return True


for n in [15, 30, 45]:
    print(f"Size: {n}")
    c1 = randint(3, size=n) + 1
    c2 = randn(n)
    for i in range(n):
        if i % 5 == 4:
            c2[i] = nan

    df = pd.DataFrame({"c1": c1, "c2": c2})
    ldf = lp.DataFrame(df)

    for skipna in [True, False]:
        for op in ops:
            print("Testing %s (skipna: %s)" % (op, skipna))
            f = getattr(pd.DataFrame, op)
            out_df = f(df, skipna=skipna)
            f = getattr(lp.DataFrame, op)
            out_ldf = f(ldf, skipna=skipna)

            out_df["c2"] = out_df["c2"].fillna(9999.0)
            out_ldf["c2"] = out_ldf["c2"].fillna(9999.0)

            assert out_ldf["c1"].equals(out_df["c1"]) and similar(
                out_ldf["c2"], out_df["c2"]
            )

    c2 = randn(n)
    for i in range(n):
        if i < (n // 2) - 1:
            c2[i] = nan

    df = pd.DataFrame({"c1": c1, "c2": c2})
    ldf = lp.DataFrame(df)

    for skipna in [True, False]:
        for op in ops:
            print("Testing %s (skipna: %s)" % (op, skipna))
            f = getattr(pd.DataFrame, op)
            out_df = f(df, skipna=skipna)
            f = getattr(lp.DataFrame, op)
            out_ldf = f(ldf, skipna=skipna)

            out_df["c2"] = out_df["c2"].fillna(9999.0)
            out_ldf["c2"] = out_ldf["c2"].fillna(9999.0)

            assert out_ldf["c1"].equals(out_df["c1"]) and similar(
                out_ldf["c2"], out_df["c2"]
            )

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
from numpy.random import permutation

from legate import pandas as lp

n = 32
df = pd.DataFrame({"c1": permutation(n) + 10, "c2": permutation(n) + 20})
ldf = lp.DataFrame(df)

print(
    f"##### df['c1'].dtype: {df['c1'].dtype}, "
    f"df['c2'].dtype: {df['c2'].dtype} #####"
)
ops = ["sum", "prod", "min", "max", "var", "std", "mean"]

for op in ops:
    print("Testing " + op)
    f = getattr(pd.DataFrame, op)
    out_df = f(df)
    f = getattr(lp.DataFrame, op)
    out_ldf = f(ldf)
    assert out_ldf.equals(out_df)

df["c2"] = df["c2"].astype("string")
ldf["c2"] = ldf["c2"].astype("string")

print(
    f"##### df['c1'].dtype: {df['c1'].dtype}, "
    f"df['c2'].dtype: {df['c2'].dtype} #####"
)

ops = ["sum", "prod", "var", "std", "mean"]

for op in ops:
    print("Testing " + op)
    f = getattr(pd.DataFrame, op)
    out_df = f(df)
    f = getattr(lp.DataFrame, op)
    out_ldf = f(ldf)
    assert out_ldf.equals(out_df)

ops = ["min", "max"]

for op in ops:
    print("Testing " + op)
    out_df = df.agg([op])
    f = getattr(lp.DataFrame, op)
    out_ldf = f(ldf)
    assert out_ldf.equals(out_df)

df["c1"] = df["c1"].astype("string")
ldf["c1"] = ldf["c1"].astype("string")

print(
    f"##### df['c1'].dtype: {df['c1'].dtype}, "
    f"df['c2'].dtype: {df['c2'].dtype} #####"
)

for op in ops:
    print("Testing " + op)
    f = getattr(pd.DataFrame, op)
    out_df = f(df)
    f = getattr(lp.DataFrame, op)
    out_ldf = f(ldf)
    assert out_ldf.equals(out_df)

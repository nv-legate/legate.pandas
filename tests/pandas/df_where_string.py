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

from legate import pandas as lp

for n in [10, 20, 40]:
    df = pd.DataFrame({"c1": list(range(n)), "c2": [1234] * n})
    df["c2"] = df["c2"].astype(pd.StringDtype())
    ldf = lp.DataFrame(df)

    out_pd = df["c2"].where(df["c1"] % 2 == 0, "Null value")
    out_lp = ldf["c2"].where(ldf["c1"] % 2 == 0, "Null value")

    assert out_lp.equals(out_pd)

    out_df = df.where(df.c1 % 2 == 0)
    out_ldf = ldf.where(ldf.c1 % 2 == 0)

    assert out_ldf["c2"].equals(out_df["c2"])

    out_pd = df["c2"].fillna("Null value")
    out_lp = ldf["c2"].fillna("Null value")

    assert out_lp.equals(out_pd)

    out_df = df.where(df.c1 % 2 == 0, df)
    out_ldf = ldf.where(ldf.c1 % 2 == 0, ldf)

    # TODO: The where function of the vanilla Pandas produces
    #       the output in AOS layout for some obscure reason,
    #       and attaching columns individually for such an output,
    #       is not gonna be fun. Here we simply make a copy of the
    #       output to force it back to SOA.

    assert out_ldf.equals(pd.DataFrame(out_df, copy=True))

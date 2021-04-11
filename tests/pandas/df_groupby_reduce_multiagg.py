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

import numpy as np
import pandas as pd
from numpy.random import permutation, randn

from legate import pandas as lp
from tests.utils import equals

n = 32
df = pd.DataFrame(
    {
        "k1": np.array(permutation(n) % 3, dtype="int64"),
        "k2": np.array(permutation(n) % 3, dtype="int64"),
        "k3": np.array(permutation(n) % 3, dtype="int64"),
        "v1": np.array(randn(n) * 100.0, dtype="int64"),
        "v2": np.array(permutation(n) % 10),
    }
)
df["v2"] = df["v2"].astype("string")
ldf = lp.DataFrame(df)

for by in [["k1"], ["k2", "k1"], ["k2", "k3"]]:
    for as_index in [True, False]:
        print(f"by={by}, as_index={as_index}, op=sum")
        assert equals(
            ldf.groupby(by=by, as_index=as_index, sort=True).sum(),
            df.groupby(by=by, as_index=as_index, sort=True).sum(),
        )
        for agg in [
            ["sum", "max"],
            {"v1": "sum", "v2": "max"},
            {"v1": ["sum", "mean"], "k1": "max"},
        ]:
            print(f"by={by}, as_index={as_index}, op={agg}")
            out_df = df.groupby(by=by, as_index=as_index, sort=True).agg(agg)
            out_ldf = ldf.groupby(by=by, as_index=as_index, sort=True).agg(agg)

            # Pandas ignores the value of as_index in fantastic ways
            # (GH #13217) and rearranging its output to match with Legate
            # is really painful. After all my attempts to make a sensible
            # transformation, I gave up and instead decided to compare on
            # the columns that appear in the Pandas output.
            assert equals(
                out_ldf, out_df, not as_index and out_df.columns.nlevels > 1
            )

df = df.set_index(["k1", "k2", "k3"])
ldf = ldf.set_index(["k1", "k2", "k3"])
for lvl in [["k1"], [1, "k1"]]:
    for as_index in [True, False]:
        print(f"level={lvl}, as_index={as_index}, op=sum")
        assert equals(
            ldf.groupby(level=lvl, as_index=as_index, sort=True).sum(),
            df.groupby(level=lvl, as_index=as_index, sort=True).sum(),
        )
        for agg in [
            ["sum", "max"],
            {"v1": "sum", "v2": "max"},
            {"v1": ["sum", "mean"], "v2": "max"},
        ]:
            print(f"level={lvl}, as_index={as_index}, op={agg}")
            out_df = df.groupby(level=lvl, as_index=as_index, sort=True).agg(
                agg
            )
            out_ldf = ldf.groupby(level=lvl, as_index=as_index, sort=True).agg(
                agg
            )

            # Pandas ignores the value of as_index in fantastic ways
            # (GH #13217) and rearranging its output to match with Legate
            # is really painful. After all my attempts to make a sensible
            # transformation, I gave up and instead decided to compare on
            # the columns that appear in the Pandas output.
            assert equals(
                out_ldf, out_df, not as_index and out_df.columns.nlevels > 1
            )

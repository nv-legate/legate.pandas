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
from tests.utils import equals

for index in [
    pd.RangeIndex(1, 21, 2),
    pd.RangeIndex(21, 1, -2),
    pd.Index(permutation(10)),
]:
    print(f"Index: {index}")
    df = pd.DataFrame(
        {
            "a": range(10),
            "b": range(1, 11),
            "c": [None if i % 4 == 1 else str(i) * 3 for i in range(10)],
            "d": [str(i % 3) for i in range(10)],
        },
        index=index,
    )
    df["c"] = df["c"].astype(pd.StringDtype())
    df["d"] = df["d"].astype("category")
    ldf = lp.DataFrame(df)

    assert equals(ldf.iloc[:, 0], df.iloc[:, 0])
    assert equals(ldf.iloc[:, [1]], df.iloc[:, [1]])
    assert equals(ldf.iloc[:, [1, 3]], df.iloc[:, [1, 3]])
    assert equals(
        ldf.iloc[:, [True, False, True, False]],
        df.iloc[:, [True, False, True, False]],
    )
    assert equals(ldf.iloc[:, 0:], df.iloc[:, 0:])
    assert equals(ldf.iloc[:, :3], df.iloc[:, :3])

    assert equals(ldf.iloc[3, 0:2].to_pandas().T.squeeze(), df.iloc[3, 0:2])

    assert equals(ldf.iloc[3:-4, 1:3], df.iloc[3:-4, 1:3])
    assert equals(ldf.iloc[-6:-4, 1:3], df.iloc[-6:-4, 1:3])
    assert equals(ldf.iloc[:-4, 1:3], df.iloc[:-4, 1:3])
    assert equals(ldf.iloc[3:, 1:3], df.iloc[3:, 1:3])

    # This should be a no-op
    ldf.iloc[0:0, [0, 1]] = -100
    df.iloc[0:0, [0, 1]] = -100

    assert equals(ldf, df)

    ldf.iloc[5, [0, 1]] = 100
    df.iloc[5, [0, 1]] = 100

    assert equals(ldf, df)

    ldf.iloc[3, [0, 1]] = ldf.iloc[3, [0, 1]] + 100
    df.iloc[3, [0, 1]] = df.iloc[3, [0, 1]] + 100

    assert equals(ldf, df)

    df.iloc[:, [0, 1]] = df.iloc[:, [0, 1]] - 100
    ldf.iloc[:, [0, 1]] = ldf.iloc[:, [0, 1]] - 100

    assert equals(ldf, df)

    sl = slice(-5, 9)
    df.iloc[sl, [0, 1]] = df.iloc[sl, [0, 1]] + 100
    ldf.iloc[sl, [0, 1]] = ldf.iloc[sl, [0, 1]] + 100

    assert equals(ldf, df)

    sl = slice(5, 8)
    df.iloc[sl, 2] = df.iloc[sl, 2].str.pad(width=9, side="both", fillchar="-")
    ldf.iloc[sl, 2] = ldf.iloc[sl, 2].str.pad(
        width=9, side="both", fillchar="-"
    )

    assert equals(ldf, df)

    sl = slice(1, 3)
    df.iloc[sl, 2] = "fill"
    ldf.iloc[sl, 2] = "fill"

    assert equals(ldf, df)

    pd_mask = (df["a"] % 3 == 0).values
    lp_mask = ldf["a"] % 3 == 0
    assert equals(ldf.iloc[lp_mask, [0, 1]], df.iloc[pd_mask, [0, 1]])

    df.iloc[pd_mask, [0, 1]] = df.iloc[pd_mask, 0:2] + 100
    ldf.iloc[lp_mask, [0, 1]] = ldf.iloc[lp_mask, 0:2] + 100

    assert equals(ldf, df)

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
from tests.utils import equals, equals_scalar

for index in [
    # pd.RangeIndex(10),
    pd.RangeIndex(1, 21, 2),
    pd.RangeIndex(21, 1, -2),
    pd.Index(permutation(10)),
]:
    print(f"Index: {index}")
    df = pd.DataFrame(
        {
            "a": range(10),
            "b": range(1, 11),
            "c": [str(i) * 3 for i in range(10)],
            "d": [str(i % 3) for i in range(10)],
        },
        index=index,
    )
    df["c"] = df["c"].astype(pd.StringDtype())
    df["d"] = df["d"].astype("category")
    ldf = lp.DataFrame(df)

    assert equals(ldf.loc[:, "a"], df.loc[:, "a"])
    assert equals(ldf.loc[:, ["b"]], df.loc[:, ["b"]])
    assert equals(ldf.loc[:, ["b", "d"]], df.loc[:, ["b", "d"]])
    assert equals(
        ldf.loc[:, [True, False, True, False]],
        df.loc[:, [True, False, True, False]],
    )
    assert equals(ldf.loc[:, "a":], df.loc[:, "a":])
    assert equals(ldf.loc[:, :"c"], df.loc[:, :"c"])

    assert equals_scalar(ldf.loc[index[0], "a"], df.loc[index[0], "a"])

    assert equals(
        ldf.loc[index[3], ["a", "b"]].to_pandas().T.squeeze(),
        df.loc[index[3], ["a", "b"]],
    )

    assert equals(
        ldf.loc[index[3] : index[-4], "b":"d"],
        df.loc[index[3] : index[-4], "b":"d"],
    )

    assert equals(ldf.loc[: index[-3], "b":"d"], df.loc[: index[-3], "b":"d"])

    assert equals(ldf.loc[index[2] :, "b":"d"], df.loc[index[2] :, "b":"d"])

    assert equals(
        ldf.loc[ldf["a"] % 3 == 0, [False, False, True, True]],
        df.loc[df["a"] % 3 == 0, [False, False, True, True]],
    )

    mask = df["a"] % 3 == 0
    assert equals(
        ldf.loc[mask, [False, False, True, True]],
        df.loc[mask, [False, False, True, True]],
    )

    mask = mask.values
    assert equals(
        ldf.loc[mask, [False, False, True, True]],
        df.loc[mask, [False, False, True, True]],
    )

    mask = list(mask)
    assert equals(
        ldf.loc[mask, [False, False, True, True]],
        df.loc[mask, [False, False, True, True]],
    )

    ldf.loc[index[3], ["a", "b"]] = ldf.loc[index[3], ["a", "b"]] - 100
    df.loc[index[3], ["a", "b"]] = df.loc[index[3], ["a", "b"]] - 100

    assert equals(ldf, df)

    mask = df["a"] % 3 == 0
    df.loc[mask, ["a", "b"]] = df.loc[mask, ["a", "b"]] + 100
    mask = ldf["a"] % 3 == 0
    ldf.loc[mask, ["a", "b"]] = ldf.loc[mask, ["a", "b"]] + 100

    assert equals(ldf, df)

    df.loc[:, ["a", "b"]] = df.loc[:, ["a", "b"]] - 100
    ldf.loc[:, ["a", "b"]] = ldf.loc[:, ["a", "b"]] - 100

    assert equals(ldf, df)

    sl = slice(index[1], index[-2])
    df.loc[sl, ["a", "b"]] = df.loc[sl, ["a", "b"]] + 100
    ldf.loc[sl, ["a", "b"]] = ldf.loc[sl, ["a", "b"]] + 100

    assert equals(ldf, df)

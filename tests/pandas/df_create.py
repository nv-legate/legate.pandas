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
from tests.utils import equals

indices = [
    pd.RangeIndex(3),
    pd.RangeIndex(1, 4),
    pd.RangeIndex(6, step=2),
    pd.RangeIndex(1, 10, step=3),
]

for index in indices:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=index)
    ldf = lp.DataFrame(df)

    # Passing Legate dataframes as arguments
    assert equals(
        lp.DataFrame(ldf, dtype="float64"), pd.DataFrame(df, dtype="float64")
    )
    assert equals(
        lp.DataFrame(ldf, columns=["a"]), pd.DataFrame(df, columns=["a"])
    )
    assert equals(
        lp.DataFrame(ldf, columns=["a"], dtype="float64"),
        pd.DataFrame(df, columns=["a"], dtype="float64"),
    )

    # Passing Legate series as arguments
    assert equals(lp.DataFrame(ldf["a"]), pd.DataFrame(df["a"]))
    assert equals(
        lp.DataFrame(ldf["a"], dtype="float32"),
        pd.DataFrame(df["a"], dtype="float32"),
    )
    assert equals(
        lp.DataFrame(ldf["a"], columns=["a"]),
        pd.DataFrame(df["a"], columns=["a"]),
    )
    sr = df["a"]
    sr.name = None
    lsr = ldf["a"]
    lsr.name = None
    assert equals(lp.DataFrame(lsr), pd.DataFrame(sr))
    assert equals(
        lp.DataFrame(lsr, columns=["A"]), pd.DataFrame(sr, columns=["A"])
    )

    # Passing Pandas dataframes as arguments
    assert equals(
        lp.DataFrame(df, dtype="float64"), pd.DataFrame(df, dtype="float64")
    )
    assert equals(
        lp.DataFrame(df, columns=["a"]), pd.DataFrame(df, columns=["a"])
    )
    assert equals(
        lp.DataFrame(df, columns=["a"], dtype="float64"),
        pd.DataFrame(df, columns=["a"], dtype="float64"),
    )

    # Passing named Legate series as arguments
    sr_a = pd.Series([1, 2, 3], index=index)
    sr_b = pd.Series([4, 5, 6], index=index)

    lsr_a = lp.Series([1, 2, 3], index=index)
    lsr_b = lp.Series([4, 5, 6], index=index)

    assert equals(
        lp.DataFrame({"a": lsr_a, "b": lsr_b}),
        pd.DataFrame({"a": sr_a, "b": sr_b}),
    )

    # Passing Pandas series as arguments
    assert equals(lp.DataFrame(df["a"]), pd.DataFrame(df["a"]))
    assert equals(
        lp.DataFrame(df["a"], dtype="float32"),
        pd.DataFrame(df["a"], dtype="float32"),
    )
    assert equals(
        lp.DataFrame(df["a"], columns=["a"]),
        pd.DataFrame(df["a"], columns=["a"]),
    )
    sr = df["a"]
    sr.name = None
    assert equals(lp.DataFrame(sr), pd.DataFrame(sr))
    assert equals(
        lp.DataFrame(sr, columns=["A"]), pd.DataFrame(sr, columns=["A"])
    )

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=index)
    ldf = lp.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=index)
    assert equals(ldf, df)


ldf1 = lp.DataFrame({1: [1, 2, 3], 2: [4, 5, 6]}, index=indices[0])
ldf2 = lp.DataFrame({1: [1, 2, 3], 2: [4, 5, 6]}, index=indices[1])

assert not ldf1.equals(ldf2)

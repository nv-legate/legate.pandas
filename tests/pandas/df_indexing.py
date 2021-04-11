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
from numpy.random import permutation

from legate import pandas as lp
from tests.utils import equals

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

    assert equals(ldf["a"], df["a"])
    assert equals(ldf[2:-2], df[2:-2])

    assert equals(ldf[["b", "c"]], df[["b", "c"]])
    assert equals(ldf[np.array(["b", "c"])], df[np.array(["b", "c"])])
    assert equals(ldf[pd.Index(["b", "c"])], df[pd.Index(["b", "c"])])

    df_ab = df[["a", "b"]]
    ldf_ab = ldf[["a", "b"]]
    pd_mask_df = df_ab % 2 == 0
    pd_mask_sr = df_ab["a"] % 2 == 0

    assert equals(ldf_ab[ldf_ab % 2 == 0], df_ab[pd_mask_df])
    assert equals(ldf_ab[pd_mask_df], df_ab[pd_mask_df])
    assert equals(ldf_ab[ldf_ab["a"] % 2 == 0], df_ab[pd_mask_sr])
    assert equals(ldf_ab[pd_mask_sr], df_ab[pd_mask_sr])
    assert equals(ldf_ab[pd_mask_sr.values], df_ab[pd_mask_sr.values])
    assert equals(ldf_ab[pd_mask_sr.to_list()], df_ab[pd_mask_sr.to_list()])

    df["e"] = df["a"]
    ldf["e"] = ldf["a"]
    assert equals(ldf, df)

    df[["e", "b"]] = df[["b", "e"]]
    ldf[["e", "b"]] = ldf[["b", "e"]]
    assert equals(ldf, df)

    df[["g", "f"]] = df[["b", "c"]]
    ldf[["g", "f"]] = ldf[["b", "c"]]
    assert equals(ldf, df)

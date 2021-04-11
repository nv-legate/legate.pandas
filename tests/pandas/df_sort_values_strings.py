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
from numpy.random import randn

from legate import pandas as lp

for n in [15, 30, 45]:
    c1 = np.array(randn(n) * 5.0, dtype=np.int64)
    c2 = np.array(randn(n) * 5.0, dtype=np.int32)
    c3 = np.array(randn(n) * 5.0, dtype=np.float64)
    c4 = np.array(randn(n) * 5.0, dtype=np.float32)

    index = pd.Index(np.arange(n) * 10)
    df = pd.DataFrame({"c1": c1, "c2": c2, "c3": c3, "c4": c4}, index=index)
    df["c1"] = df["c1"].astype(pd.StringDtype())
    df["c3"] = df["c3"].astype(pd.StringDtype())
    ldf = lp.DataFrame(df)

    for ascending in [True, False]:
        for ignore_index in [False, True]:
            print(
                "Size: %u, Ascending: %s, Ignore index: %s"
                % (n, ascending, ignore_index)
            )

            out_df = df.sort_values(
                ["c1", "c2", "c3"],
                ignore_index=ignore_index,
                ascending=ascending,
            )
            out_ldf = ldf.sort_values(
                ["c1", "c2", "c3"],
                ignore_index=ignore_index,
                ascending=ascending,
            )

            assert out_ldf.equals(out_df)

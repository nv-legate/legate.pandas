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

for n in [15, 30, 50]:
    c1 = np.array(permutation(n), dtype=np.int64) % 5
    c2 = np.array(randn(n) * 5.0, dtype=np.int32)
    c3 = np.array(randn(n) * 5.0, dtype=np.float64)
    c4 = np.array(randn(n) * 5.0, dtype=np.float32)

    df = pd.DataFrame({"c1": c1, "c2": c2, "c3": c3, "c4": c4})
    ldf = lp.DataFrame(df)

    for ascending in [True, False]:
        print("Size: %u, Ascending: %s" % (n, ascending))

        out_df = df.set_index("c1")
        out_ldf = ldf.set_index("c1")

        out_df = out_df.sort_index(ascending=ascending)
        out_ldf = out_ldf.sort_index(ascending=ascending)

        assert out_ldf.equals(out_df)

    for level, ascending in zip([0, 1, [0, 1]], [False, True, [True, False]]):
        print("Size: %u, Level: %s, Ascending: %s" % (n, level, ascending))

        out_df = df.set_index(["c1", "c2"])
        out_ldf = ldf.set_index(["c1", "c2"])

        out_df = out_df.sort_index(level=level, ascending=ascending)
        out_df = out_df.sort_index(level=level, ascending=ascending)
        out_ldf = out_ldf.sort_index(level=level, ascending=ascending)

        assert out_ldf.equals(out_df)
    break

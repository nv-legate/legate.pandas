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

from legate import pandas as lp

for n in [10, 20, 40]:
    c2 = [1] * n
    for tpl in [(0, n, 1), (5, n + 5, 1), (2, 2 * n + 2, 2)]:
        for i in range(n):
            if i % 2 == 0:
                c2[i] = np.nan

        (start, stop, step) = tpl
        df = pd.DataFrame(
            {"c1": list(range(n)), "c2": pd.Series(c2, dtype=np.float64)},
            index=pd.RangeIndex(start=start, stop=stop, step=step),
        )
        ldf = lp.DataFrame(df)

        df = df[df.c1 % 4 == 0].fillna(9999.0)
        ldf = ldf[ldf.c1 % 4 == 0].fillna(9999.0)

        assert ldf.equals(df)

        df = df[df.c1 % 3 == 1]
        ldf = ldf[ldf.c1 % 3 == 1]

        assert ldf.equals(df)

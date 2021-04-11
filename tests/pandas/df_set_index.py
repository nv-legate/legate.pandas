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

for drop in [True, False]:
    for append in [False, True]:
        for index in [
            pd.RangeIndex(3),
            pd.MultiIndex.from_arrays(
                [[10, 20, 30], [11, 21, 31]], names=("A", "B")
            ),
        ]:
            df = pd.DataFrame(
                {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, index=index
            )
            ldf = lp.DataFrame(df)

            out_df = df.set_index("a", drop=drop, append=append)
            out_ldf = ldf.set_index("a", drop=drop, append=append)

            assert out_ldf.equals(out_df)

            out_df = df.set_index(["a", "c"], drop=drop, append=append)
            out_ldf = ldf.set_index(["a", "c"], drop=drop, append=append)

            assert out_ldf.equals(out_df)

            idx = [100, 200, 300]
            out_df = df.set_index(["a", idx], drop=drop, append=append)
            out_ldf = ldf.set_index(["a", idx], drop=drop, append=append)

            assert out_ldf.equals(out_df)

            idx = pd.Series([100, 200, 300], name="new_idx")
            out_df = df.set_index(["a", idx], drop=drop, append=append)
            out_ldf = ldf.set_index(["a", idx], drop=drop, append=append)

            assert out_ldf.equals(out_df)

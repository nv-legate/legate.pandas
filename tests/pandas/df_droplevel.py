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

df = pd.DataFrame(
    {
        (1, "a", "A"): [1, 1, 1, 2, 3, 3],
        (2, "b", "B"): [1, 2, 3, 1, 2, 3],
        (3, "c", "C"): [1, 1, 1, 2, 2, 2],
    },
    index=pd.MultiIndex.from_arrays(
        [[1, 2, 3, 4, 5, 6], [1, 1, 2, 2, 3, 3], [5, 4, 3, 2, 1, 0]]
    ).rename(("idx_1", "idx_2", "idx_3")),
)
df.columns.names = ("col_1", "col_2", "col_3")
ldf = lp.DataFrame(df)

assert equals(ldf.droplevel(0, axis=0), df.droplevel(0, axis=0))
assert equals(ldf.droplevel("idx_2", axis=0), df.droplevel("idx_2", axis=0))
assert equals(
    ldf.droplevel(["idx_1", 2], axis=0), df.droplevel(["idx_1", 2], axis=0)
)

assert equals(ldf.droplevel(0, axis=1), df.droplevel(0, axis=1))
assert equals(ldf.droplevel("col_2", axis=1), df.droplevel("col_2", axis=1))
assert equals(
    ldf.droplevel(["col_1", 2], axis=1), df.droplevel(["col_1", 2], axis=1)
)

assert equals(
    ldf.iloc[:, 0].droplevel(0, axis=0), df.iloc[:, 0].droplevel(0, axis=0)
)
assert equals(
    ldf.iloc[:, 0].droplevel("idx_2", axis=0),
    df.iloc[:, 0].droplevel("idx_2", axis=0),
)
assert equals(
    ldf.iloc[:, 0].droplevel(["idx_1", 2], axis=0),
    df.iloc[:, 0].droplevel(["idx_1", 2], axis=0),
)

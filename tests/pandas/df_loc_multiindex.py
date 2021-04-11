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
from tests.utils import equals, equals_scalar, must_fail


def _test(ex, df, *args):
    def _loc():
        df.loc[args]

    must_fail(ex, _loc)


index = pd.MultiIndex.from_arrays(
    [
        [3, 2, 1, 0, 3, 2, 1, 0, 4, 3],
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2],
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
    ]
)

df = pd.DataFrame({"a": range(10)}, index=index)
ldf = lp.DataFrame(df)

_test(KeyError, ldf, (2, 0, 3))

assert equals_scalar(ldf.loc[(2, 0, 1), "a"], df.loc[(2, 0, 1), "a"])
assert equals(ldf.loc[(2, 0), "a"], df.loc[(2, 0), "a"])
assert equals(ldf.loc[2, "a"], df.loc[2, "a"])
assert equals(ldf.loc[4, "a"], df.loc[4, "a"])

ldf.loc[(2, 0), "a"] = 100
df.loc[(2, 0), "a"] = 100
assert equals(ldf, df)

ldf.loc[4, "a"] = 200
df.loc[4, "a"] = 200
assert equals(ldf, df)

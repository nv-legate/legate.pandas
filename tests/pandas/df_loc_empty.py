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
from tests.utils import equals, must_fail


def _test(ex, df, *args):
    def _loc():
        df.loc[args]

    must_fail(ex, _loc)


n = 17

for index in [pd.RangeIndex(3, n + 3), pd.Index(list(range(3, n + 3)))]:
    df_copy = lp.DataFrame({"a": range(n)}, index=index)
    df = lp.DataFrame({"a": range(n)}, index=index)

    _test(KeyError, df, n + 3)
    _test(KeyError, df, n + 4, "a")

    assert len(df.loc[n + 3 : n + 4]) == 0

    df.loc[n + 3] = 100
    assert equals(df_copy, df)

    df.loc[n + 3 : n + 4] = 200
    assert equals(df_copy, df)

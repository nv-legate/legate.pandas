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


def _test(ex, sr, *args):
    def _loc():
        sr.loc[args]

    must_fail(ex, _loc)


n = 17

for index in [pd.RangeIndex(3, n + 3), pd.Index(list(range(3, n + 3)))]:
    sr_copy = lp.Series(range(n), index=index)
    sr = lp.Series(range(n), index=index)

    _test(KeyError, sr, n + 3)

    assert len(sr.loc[n + 3 : n + 4]) == 0

    sr.loc[n + 3] = 100
    assert equals(sr_copy, sr)

    sr.loc[n + 3 : n + 4] = 200
    assert equals(sr_copy, sr)

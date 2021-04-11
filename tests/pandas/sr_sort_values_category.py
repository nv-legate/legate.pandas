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

for n in [15, 30, 45]:
    codes = ["%03d" % (i % 4) for i in range(n)]
    s = pd.Series(codes, index=pd.RangeIndex(2 * n, 0, -2)).astype("category")
    for i in range(n):
        if i % 4 == 2:
            s[i] = None
    ls = lp.Series(s)

    for ascending in [True, False]:
        # XXX: We only test with ignore_index being True, as Pandas does not
        #      stable sort the index when the key is a categorical column.
        ignore_index = True
        print(
            f"Size: {n}, Ascending: {ascending}, Ignore index: {ignore_index}"
        )

        out_pd = s.sort_values(ascending=ascending, ignore_index=ignore_index)
        out_lp = ls.sort_values(ascending=ascending, ignore_index=ignore_index)

        assert equals(out_lp, out_pd)

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
from tests.utils import equals

for n in [15, 30, 45]:
    c = np.array(randn(n) * 5.0, dtype=np.float32)
    for i in range(n):
        if i % 4 == 1:
            c[i] = np.nan

    s = pd.Series(c)
    ls = lp.Series(s)

    for ascending in [True, False]:
        for na_position in ["first", "last"]:
            print(
                f"Size: {n}, Ascending: {ascending}, "
                f"Na_position: {na_position}"
            )

            out_pd = s.sort_values(
                ascending=ascending, na_position=na_position
            )
            out_lp = ls.sort_values(
                ascending=ascending, na_position=na_position
            )

            assert equals(out_lp, out_pd)

for n in [15, 30, 45]:
    codes = ["%03d" % (i % 4) for i in range(n)]
    for i in range(n):
        if i % 5 == 2:
            codes[i] = None
    s = pd.Series(codes, index=pd.RangeIndex(2 * n, 0, -2), dtype="string")
    ls = lp.Series(s)

    for ascending in [True, False]:
        for na_position in ["first", "last"]:
            print(
                f"Size: {n}, Ascending: {ascending}, "
                f"Na_position: {na_position}"
            )

            # XXX: We set ignore_index to True, as Pandas does not stable sort
            # the index when the key is a string column.
            out_pd = s.sort_values(
                ignore_index=True, ascending=ascending, na_position=na_position
            )
            out_lp = ls.sort_values(
                ignore_index=True, ascending=ascending, na_position=na_position
            )

            assert equals(out_lp, out_pd)

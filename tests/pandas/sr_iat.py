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
from numpy.random import permutation

from legate import pandas as lp
from tests.utils import equals_scalar, must_fail


def _test(ex, sr, *args):
    def _make_access():
        sr.iat[args]

    must_fail(ex, _make_access)


for index in [
    pd.RangeIndex(10),
    pd.RangeIndex(1, 21, 2),
    pd.RangeIndex(21, 1, -2),
    pd.Index(permutation(10)),
]:
    print(f"Index: {index}")
    sr = pd.Series(range(10), index=index)
    lsr = lp.Series(sr)

    for idx in range(3, 8):
        print(f"Testing lsr.at[{idx}].__getitem__")
        out_pd = sr.iat[idx]
        out_lp = lsr.iat[idx]
        assert equals_scalar(out_lp, out_pd)

    for idx, val in enumerate([100, 200, 300]):
        print(f"Testing lsr.at[{idx + 3}].__setitem__")
        sr.iat[idx + 3] = val
        lsr.iat[idx + 3] = val

        out_pd = sr.iat[idx + 3]
        out_lp = lsr.iat[idx + 3]
        assert equals_scalar(out_lp, out_pd)

# Negative tests

lsr = lp.Series(range(10))
_test(ValueError, lsr, slice(None))

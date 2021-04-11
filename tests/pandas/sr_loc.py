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
from tests.utils import equals, equals_scalar

for index in [
    pd.RangeIndex(1, 21, 2),
    pd.RangeIndex(21, 1, -2),
    pd.Index(permutation(10)),
]:
    print(f"Index: {index}")
    sr = pd.Series(range(10), index=index)
    lsr = lp.Series(range(10), index=index)

    assert equals(lsr.loc[:], sr.loc[:])
    assert equals_scalar(lsr.loc[index[0]], sr.loc[index[0]])

    assert equals(lsr.loc[index[3] : index[-4]], sr.loc[index[3] : index[-4]])
    assert equals(lsr.loc[: index[-3]], sr.loc[: index[-3]])
    assert equals(lsr.loc[index[2] :], sr.loc[index[2] :])

    pd_mask = sr % 3 == 0
    lp_mask = mask = lsr % 3 == 0

    for mask in [pd_mask, pd_mask.values, pd_mask.to_list(), lp_mask]:
        assert equals(lsr.loc[mask], sr.loc[pd_mask])

    sr.loc[pd_mask] = sr.loc[pd_mask] + 100
    lsr.loc[lp_mask] = lsr.loc[lp_mask] + 100

    assert equals(lsr, sr)

    sr.loc[:] = sr.loc[:] - 100
    lsr.loc[:] = lsr.loc[:] - 100

    assert equals(lsr, sr)

    sl = slice(index[1], index[-2])
    sr.loc[sl] = sr.loc[sl] + 100
    lsr.loc[sl] = lsr.loc[sl] + 100

    assert equals(lsr, sr)

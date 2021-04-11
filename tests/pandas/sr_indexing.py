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

    assert equals(lsr[:], sr[:])
    assert equals_scalar(lsr[index[3]], sr[index[3]])

    assert equals(lsr[3:-4], sr[3:-4])
    assert equals(lsr[:-3], sr[:-3])
    assert equals(lsr[2:], sr[2:])

    pd_mask = sr % 3 == 0
    lp_mask = mask = lsr % 3 == 0

    for mask in [pd_mask, pd_mask.values, pd_mask.to_list(), lp_mask]:
        assert equals(lsr[mask], sr[pd_mask])

    sr[pd_mask] = sr[pd_mask] + 100
    lsr[lp_mask] = lsr[lp_mask] + 100

    assert equals(lsr, sr)

    sr[:] = sr[:] - 100
    lsr[:] = lsr[:] - 100

    assert equals(lsr, sr)

    sl = slice(1, -2)
    sr[sl] = sr[sl] + 100
    lsr[sl] = lsr[sl] + 100

    assert equals(lsr, sr)

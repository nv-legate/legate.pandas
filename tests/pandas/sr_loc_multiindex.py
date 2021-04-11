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


def _test(ex, sr, *args):
    def _loc():
        sr.loc[args]

    must_fail(ex, _loc)


index = pd.MultiIndex.from_arrays(
    [
        [3, 2, 1, 0, 3, 2, 1, 0, 4, 3],
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2],
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
    ]
)

sr = pd.Series(range(10), index=index)
lsr = lp.Series(sr)

_test(KeyError, lsr, (2, 0, 3))

assert equals_scalar(
    lsr.loc[
        (2, 0, 1),
    ],
    sr.loc[
        (2, 0, 1),
    ],
)
assert equals(
    lsr.loc[
        (2, 0),
    ],
    sr.loc[
        (2, 0),
    ],
)
assert equals(lsr.loc[2], sr.loc[2])
assert equals(lsr.loc[4], sr.loc[4])

lsr.loc[
    (3, 0, 0),
] = 50
sr.loc[(3, 0, 0)] = 50
assert equals(lsr, sr)

lsr.loc[
    (2, 0),
] = 100
sr.loc[(2, 0)] = 100
assert equals(lsr, sr)

lsr.loc[
    4,
] = 200
sr.loc[
    4,
] = 200
assert equals(lsr, sr)

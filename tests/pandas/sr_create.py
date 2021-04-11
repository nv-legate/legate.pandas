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


def equals(a, b):
    result = a.equals(b)
    if not result:
        print(a)
        print(b)
    return result


indices = [
    pd.RangeIndex(1, 4),
    pd.RangeIndex(9, step=2),
    pd.RangeIndex(1, 20, step=3),
]

for index in indices:
    sr = pd.Series(list(range(len(index))), index=index)
    lsr = lp.Series(sr)

    # Passing Legate series as arguments
    assert equals(lp.Series(lsr), pd.Series(sr))
    assert equals(
        lp.Series(lsr, dtype="float32"),
        pd.Series(sr, dtype="float32"),
    )
    assert equals(lp.Series(lsr, name="A"), pd.Series(sr, name="A"))

    # Passing Pandas series as arguments
    assert equals(lp.Series(sr), pd.Series(sr))
    assert equals(
        lp.Series(sr, dtype="float32"),
        pd.Series(sr, dtype="float32"),
    )
    assert equals(lp.Series(sr, name="A"), pd.Series(sr, name="A"))

lsr1 = lp.Series(list(range(len(indices[0]))), index=indices[0])
lsr2 = lp.Series(list(range(len(indices[1]))), index=indices[1])

assert not lsr1.equals(lsr2)

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
from numpy.random import permutation, randn

from legate import pandas as lp
from tests.utils import equals

n = 10
sr = pd.Series(
    np.array(randn(1, n)[0] * 100.0, dtype="int64"),
    name="A",
    index=pd.MultiIndex.from_arrays(
        [
            np.array(permutation(n), dtype="int64"),
            np.array(permutation(n), dtype="int64"),
            np.array(permutation(n), dtype="int64"),
        ],
        names=("k1", "k2", "k3"),
    ),
)
lsr = lp.Series(sr)

for lvl in [["k1"], [1, "k1"], [0, 2]]:
    print(f"level={lvl}, op=sum")
    out_sr = sr.groupby(level=lvl, sort=True).sum()
    out_lsr = lsr.groupby(level=lvl, sort=True).sum()
    assert equals(out_lsr, out_sr)

    agg = ["sum", "max"]
    print(f"level={lvl}, op={agg}")
    out_sr = sr.groupby(level=lvl, sort=True).agg(agg)
    out_lsr = lsr.groupby(level=lvl, sort=True).agg(agg)

    # Pandas ignores the value of as_index in fantastic ways
    # (GH #13217) and rearranging its output to match with Legate
    # is really painful. After all my attempts to make a sensible
    # transformation, I gave up and instead decided to compare on
    # the columns that appear in the Pandas output.
    assert equals(out_lsr, out_sr)

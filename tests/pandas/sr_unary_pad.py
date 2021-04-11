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
from numpy.random import randint

from legate import pandas as lp

n = 23
sides = ["left", "right", "both"]

for side in sides:
    strs = [
        "".join(
            [
                chr(ord("A") + c) if c < 23 else chr(ord("a") + (c - 23))
                for c in randint(0, 46, rng)
            ]
        )
        for rng in randint(1, 15, n)
    ]
    s = pd.Series(strs, dtype=pd.StringDtype())
    ls = lp.Series(s)
    out_s = s.str.pad(width=10, side=side, fillchar="-")
    out_ls = ls.str.pad(width=10, side=side, fillchar="-")
    assert out_ls.equals(out_s)

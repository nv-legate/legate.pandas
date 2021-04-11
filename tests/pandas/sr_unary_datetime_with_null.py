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

from legate import pandas as lp

start = pd.to_datetime("2015-01-01")
end = pd.to_datetime("2020-01-01")

start_u = start.value // 10 ** 9
end_u = end.value // 10 ** 9

n = 30
s = pd.Series(
    10 ** 9 * np.random.randint(start_u, end_u, n, dtype=np.int64),
    dtype=np.dtype("datetime64[ns]"),
)
for i in range(n):
    if i % 3 == 0:
        s[i] = pd.NaT
ls = lp.Series(s)

fields = ["year", "month", "day", "hour", "minute", "second", "weekday"]

for field in fields:
    print("Testing " + field)
    out_s = getattr(s.dt, field).fillna(0.0)
    out_ls = getattr(ls.dt, field).fillna(0).astype(np.double)
    assert out_ls.equals(out_s)

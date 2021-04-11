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

n = 17

a = [str(i) * 3 for i in range(n)]

for i in range(n):
    if i % 4 == 0:
        a[i] = None

s = pd.Series(a)
s = s.astype(pd.StringDtype())

ls = lp.DataFrame(s)

out_pd = s.dropna()
out_lp = ls.dropna()
assert out_lp.equals(out_pd)

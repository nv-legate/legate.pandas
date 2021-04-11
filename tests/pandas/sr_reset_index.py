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

sr = pd.Series([1, 2, 3])
lsr = lp.Series(sr)

out_pd = sr.reset_index()
out_lp = lsr.reset_index()

assert out_lp.equals(out_pd)

sr = pd.Series([1, 2, 3])
lsr = lp.Series(sr)

out_pd = sr.reset_index(name="idx")
out_lp = lsr.reset_index(name="idx")

assert out_lp.equals(out_pd)

sr = pd.Series([1, 2, 3])
sr.index.name = "test"
lsr = lp.Series(sr)

out_pd = sr.reset_index()
out_lp = lsr.reset_index()

assert out_lp.equals(out_pd)

sr = pd.Series([1, 2, 3], index=pd.Index([10, 20, 30]))
lsr = lp.Series(sr)

sr.reset_index(drop=True, inplace=True)
lsr.reset_index(drop=True, inplace=True)

assert lsr.equals(sr)

sr = pd.Series(
    [1, 2, 3],
    index=pd.MultiIndex.from_tuples([(10, 20), (11, 21), (12, 22)]),
)
lsr = lp.Series(sr)

out_pd = sr.reset_index()
out_lp = lsr.reset_index()

assert out_lp.equals(out_pd)

sr = pd.Series(
    [1, 2, 3],
    index=pd.MultiIndex.from_tuples(
        [(10, 20, 30), (11, 21, 31), (12, 22, 32)]
    ),
)
sr.index.names = [None, "test", "test2"]
lsr = lp.Series(sr)

assert lsr.reset_index().equals(sr.reset_index())
assert lsr.reset_index(level=1).equals(sr.reset_index(level=1))
assert lsr.reset_index(level=[0, 2]).equals(sr.reset_index(level=[0, 2]))

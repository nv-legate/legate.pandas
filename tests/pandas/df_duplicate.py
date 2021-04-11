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

df = pd.DataFrame({"c1": [1, 2, 3], "c2": [4, 5, 6], "c3": [7, 8, 9]})
ldf = lp.DataFrame(df)

df = df.rename(columns={"c3": "c1"}, copy=False)
ldf = ldf.rename(columns={"c3": "c1"}, copy=False)

assert len(ldf[["c1"]].columns) == 2
assert ldf.equals(df)
assert ldf[["c1", "c2"]].equals(df[["c1", "c2"]])
assert (ldf[["c1", "c2"]] + ldf).equals(df[["c1", "c2"]] + df)


del df["c1"]
del ldf["c1"]

assert len(ldf.columns) == 1
assert ldf.equals(df)

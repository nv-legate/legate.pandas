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

df = pd.DataFrame(
    {
        "a": range(10),
        "b": range(1, 11),
        "c": [str(i) * 3 for i in range(10)],
        "d": [str(i % 3) for i in range(10)],
    }
)
df["c"] = df["c"].astype(pd.StringDtype())
df["d"] = df["d"].astype("category")
ldf = lp.DataFrame(df)

assert ldf.tail(2).equals(df.tail(2))
assert ldf.tail(9).equals(df.tail(9))

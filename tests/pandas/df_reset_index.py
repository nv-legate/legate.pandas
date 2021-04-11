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

df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
ldf = lp.DataFrame(df)

df = df.reset_index()
ldf = ldf.reset_index()

assert ldf.equals(df)

df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
df.index.name = "test"
ldf = lp.DataFrame(df)

df = df.reset_index()
ldf = ldf.reset_index()

assert ldf.equals(df)

df = pd.DataFrame(
    {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]},
    index=pd.MultiIndex.from_tuples([(10, 20), (11, 21), (12, 22)]),
)
ldf = lp.DataFrame(df)

df = df.reset_index()
ldf = ldf.reset_index()

assert ldf.equals(df)

df = pd.DataFrame(
    {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]},
    index=pd.MultiIndex.from_tuples(
        [(10, 20, 30), (11, 21, 31), (12, 22, 32)]
    ),
)
df.index.names = [None, "test", "test2"]
ldf = lp.DataFrame(df)

assert ldf.reset_index().equals(df.reset_index())
assert ldf.reset_index(level=1).equals(df.reset_index(level=1))
assert ldf.reset_index(level=[0, 2]).equals(df.reset_index(level=[0, 2]))
assert ldf.reset_index(level=[2, 0]).equals(df.reset_index(level=[2, 0]))

df = pd.DataFrame(
    {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]},
    index=pd.MultiIndex.from_tuples(
        [(10, 20, 30), (11, 21, 31), (12, 22, 32)]
    ),
)
df.index.names = [None, "test", "test2"]
df.columns = pd.MultiIndex.from_arrays([df.columns, ["A", "B", "C"]])
ldf = lp.DataFrame(df)

assert ldf.reset_index(level=1).equals(df.reset_index(level=1))
assert ldf.reset_index(level=[0, 1], col_level=1, col_fill="idx").equals(
    df.reset_index(level=[0, 1], col_level=1, col_fill="idx")
)
assert ldf.reset_index(
    level=[None, "test2"], col_level=1, col_fill="idx"
).equals(df.reset_index(level=[None, "test2"], col_level=1, col_fill="idx"))

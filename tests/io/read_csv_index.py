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

import os
from collections import OrderedDict

import pandas as pd

from legate import pandas as lp

path = os.path.join(os.path.dirname(__file__), "files", "read_csv_index.csv")
names = ["__lvl1__", "__lvl2__", "a", "b"]
dtypes = OrderedDict(
    [
        ("__lvl1__", "int64"),
        ("__lvl2__", "float64"),
        ("a", "int64"),
        ("b", "float64"),
    ]
)

df = pd.read_csv(path, names=names, dtype=dtypes, index_col=[1, 0])
ldf = lp.read_csv(path, names=names, dtype=dtypes, index_col=[1, 0])
assert ldf.equals(df)

df = pd.read_csv(
    path, names=names, dtype=dtypes, index_col=["__lvl1__", "__lvl2__"]
)
ldf = lp.read_csv(
    path, names=names, dtype=dtypes, index_col=["__lvl1__", "__lvl2__"]
)
assert ldf.equals(df)

path = os.path.join(
    os.path.dirname(__file__), "files", "read_csv_index_and_header.csv"
)

df = pd.read_csv(path, index_col=[1, 0])
ldf = lp.read_csv(path, index_col=[1, 0])

assert ldf.equals(df)

df = pd.read_csv(
    path, names=names, header=0, index_col=["__lvl2__", "__lvl1__"]
)
ldf = lp.read_csv(
    path, names=names, header=0, index_col=["__lvl2__", "__lvl1__"]
)

assert ldf.equals(df)

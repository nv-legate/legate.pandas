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

import itertools
import os
from collections import OrderedDict

import pandas as pd

from legate import pandas as lp

base = os.path.join(os.path.dirname(__file__), "files")
paths1 = [
    os.path.join(base, "read_csv_datetime.csv"),
    os.path.join(base, "read_csv_datetime.csv.gz"),
    os.path.join(base, "read_csv_datetime.csv.bz2"),
]
paths2 = [
    os.path.join(base, "read_csv_category.csv"),
    os.path.join(base, "read_csv_category.csv.gz"),
    os.path.join(base, "read_csv_category.csv.bz2"),
]

names = ["a", "b"]
dtypes = OrderedDict([("a", "int64"), ("b", str)])

for path1, path2 in itertools.product(paths1, paths2):
    print(f"{path1} {path2}")
    df = pd.concat(
        [
            pd.read_csv(path1, names=names, dtype=dtypes, index_col=False),
            pd.read_csv(path2, names=names, dtype=dtypes, index_col=False),
        ],
        ignore_index=True,
    )

    ldf = lp.read_csv(
        [path1, path2], names=names, dtype=dtypes, index_col=False
    )

    assert ldf.equals(df)

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

import pandas as pd

from legate import pandas as lp

names = ["a", "b"]
paths = [
    os.path.join(os.path.dirname(__file__), "files", "read_csv_quotes.csv"),
    os.path.join(os.path.dirname(__file__), "files", "read_csv_colons.csv"),
]
quotechars = ['"', ":"]
for path, quotechar in zip(paths, quotechars):
    df = pd.read_csv(
        path,
        names=names,
        dtype="string",
        quotechar=quotechar,
        skipfooter=1,
        engine="python",
    )
    ldf = lp.read_csv(
        path, names=names, dtype="string", quotechar=quotechar, skipfooter=1
    )
    assert ldf.equals(df)

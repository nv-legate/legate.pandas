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

s = [
    "1900-01-01",
    "1900-01-02",
    "1900-01-03",
    "1900-01-04",
    "1900-01-05",
    "1900-01-06",
    "1900-01-07",
    "1900-01-08",
    "1900-01-09",
    "1900-01-10",
    "1900-01-11",
    "1900-01-12",
    "1900-01-13",
    "1900-01-14",
    "1900-01-15",
    "1900-01-16",
    "1900-01-17",
    "1900-01-18",
    "1900-01-19",
    "1900-01-20",
    "1900-01-21",
    "1900-01-22",
    "1900-01-23",
]
s = pd.Series(s, dtype=pd.StringDtype())
ls = lp.Series(s)

out_s = pd.to_datetime(s, format="%Y-%m-%d")
out_ls = lp.to_datetime(ls, format="%Y-%m-%d")

assert out_ls.equals(out_s)

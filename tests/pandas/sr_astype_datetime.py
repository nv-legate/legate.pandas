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

s = [
    "2000-01-01",
    "2000-01-02",
    "2000-01-03",
    "2000-01-04",
    "2000-01-05",
    "2000-01-06",
    "2000-01-07",
    "2000-01-08",
    "2000-01-09",
    "2000-01-10",
    "2000-01-11 11:00:00",
    "2000-01-12",
    "2000-01-13",
    "2000-01-14",
    "2000-01-15",
    "2000-01-16",
    "2000-01-17",
    "2000-01-18",
    "2000-01-19",
    "2000-01-20",
    "2000-01-21",
    "2000-01-22",
    "2000-01-23",
]
s = pd.Series(s, dtype=pd.StringDtype())
ls = lp.Series(s)

date_s = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S")
date_ls = lp.to_datetime(ls, format="%Y-%m-%d %H:%M:%S")

int_s = date_s.astype(np.int64)
int_ls = date_ls.astype(np.int64)

assert int_ls.equals(int_s)

str_s = date_s.astype("string")
str_ls = date_ls.astype("string")

assert str_ls.equals(str_s)

date_s = str_s.astype("datetime64[ns]")
date_ls = str_ls.astype("datetime64[ns]")

assert date_ls.equals(date_s)

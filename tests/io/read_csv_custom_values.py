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

path = os.path.join(
    os.path.dirname(__file__), "files", "read_csv_custom_values.csv"
)
names = ["a", "b"]
dtypes = OrderedDict([("a", "bool"), ("b", "float64")])

true_values = ["this is true", "this is also True"]
false_values = ["this is false", "this is also FALSE"]
na_values = ["this is null", "this is NA"]

df = pd.read_csv(
    path,
    names=names,
    dtype=dtypes,
    true_values=true_values,
    false_values=false_values,
    na_values=na_values,
)
ldf = lp.read_csv(
    path,
    names=names,
    dtype=dtypes,
    true_values=true_values,
    false_values=false_values,
    na_values=na_values,
)

assert ldf.equals(df)

df = pd.read_csv(
    path,
    names=names,
    dtype=dtypes,
    true_values=true_values,
    false_values=false_values,
    na_values=na_values,
    nrows=3,
)
ldf = lp.read_csv(
    path,
    names=names,
    dtype=dtypes,
    true_values=true_values,
    false_values=false_values,
    na_values=na_values,
    nrows=3,
)

assert ldf.equals(df)

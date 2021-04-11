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
from tests.utils import equals, must_fail


def _test(ex, df, *args):
    must_fail(ex, df.insert, *args)


df = pd.DataFrame()
ldf = lp.DataFrame()

df.insert(0, "a", 1)
ldf.insert(0, "a", 1)

assert equals(ldf, df)

df = pd.DataFrame(index=[1, 2, 3])
ldf = lp.DataFrame(index=[1, 2, 3])

df.insert(0, "a", 1)
ldf.insert(0, "a", 1)

assert equals(ldf, df)

df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
ldf = lp.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

df.insert(0, "c", 1)
ldf.insert(0, "c", 1)

assert equals(ldf, df)

df.insert(2, "d", 2)
ldf.insert(2, "d", 2)

assert equals(ldf, df)

df.insert(1, "e", df["b"])
ldf.insert(1, "e", ldf["b"])

assert equals(ldf, df)

_test(TypeError, ldf, "a", "a", 1)
_test(ValueError, ldf, -1, "a", 1)
_test(IndexError, ldf, 6, "a", 1)
_test(ValueError, ldf, 5, "a", 1)
_test(ValueError, ldf, 5, "f", ldf)

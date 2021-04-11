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
from tests.utils import equals

df = pd.DataFrame()
ldf = lp.DataFrame()

assert equals(ldf, df)

df["a"] = 0
ldf["a"] = 0

assert equals(ldf, df)

df = pd.DataFrame(columns=["a", "b"])
ldf = lp.DataFrame(columns=["a", "b"])

assert equals(ldf, df)

df.loc[:, "a"] = "1"
ldf.loc[:, "a"] = "1"

assert equals(ldf, df)

df = pd.DataFrame(index=pd.Index([1, 2, 3]))
ldf = lp.DataFrame(index=pd.Index([1, 2, 3]))

df[["a", "b"]] = 1
ldf[["a", "b"]] = 1

assert equals(ldf, df)

df = pd.DataFrame(index=pd.Index([1, 2, 3]), columns=["a", "b", "c"])
ldf = lp.DataFrame(index=pd.Index([1, 2, 3]), columns=["a", "b", "c"])

df["a"] = 1
ldf["a"] = 1

assert equals(ldf, df)

df[["b", "c"]] = "abcd"
ldf[["b", "c"]] = "abcd"

assert equals(ldf, df)

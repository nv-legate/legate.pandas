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
from numpy.random import randint

from legate import pandas as lp

n = 30
df = pd.DataFrame({"c0": randint(0, 100, n, dtype=np.int64)})
ldf = lp.DataFrame(df)

df["c1"] = pd.to_datetime("2021-01-01")
ldf["c1"] = lp.to_datetime("2021-01-01")

assert ldf.equals(df)

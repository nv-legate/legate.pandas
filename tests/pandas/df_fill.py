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

val_dtypes = [np.int32, np.float32, np.int64, np.float64]

for i in range(len(val_dtypes)):
    val = randint(0, 100, 1)[0]
    df["c" + str(i + 1)] = val_dtypes[i](val)
    ldf["c" + str(i + 1)] = val_dtypes[i](val)

assert ldf.equals(df)

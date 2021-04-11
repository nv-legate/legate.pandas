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
from tests.utils import equals

ops = [
    "add",
    "sub",
    "mul",
    "div",
    "truediv",
    "floordiv",
    "mod",
    # TODO: nans_to_nulls is required to match the pandas result
    # "pow",
    "lt",
    "gt",
    "le",
    "ge",
    "ne",
]

val_dtypes = [
    np.int16,
    np.int32,
    np.float32,
    np.int64,
    np.float64,
]

n = 17


for dtype in val_dtypes:
    df1 = pd.DataFrame(
        {
            1: np.array(randint(1, 100, n), dtype=dtype),
            5: np.array(randint(1, 100, n), dtype=dtype),
        }
    )
    ldf1 = lp.DataFrame(df1)
    df2 = pd.DataFrame(
        {
            1: np.array(randint(1, 100, n), dtype=dtype),
            5: np.array(randint(1, 100, n), dtype=dtype),
        }
    )
    ldf2 = lp.DataFrame(df2)
    for op in ops:
        print("Testing " + op + " with operands of type " + str(dtype))
        f = getattr(pd.DataFrame, op)
        out_pd = f(df1, 2)
        f = getattr(lp.DataFrame, op)
        out_lp = f(ldf1, 2)
        assert equals(out_lp, out_pd)

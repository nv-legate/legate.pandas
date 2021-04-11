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

n = 14

a = [randint(0, 5) for _ in range(n)]
s = pd.Series(a)
ls = lp.Series(s)

dtypes = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float32,
    np.float64,
    bool,
]

for dtype in dtypes:
    print(f"Testing string conversions to and from {dtype}")
    out_s = s.astype(dtype).astype(str)
    out_ls = ls.astype(dtype).astype(str)
    assert equals(out_ls, out_s)

    out_s = out_s.astype(dtype)
    out_ls = out_ls.astype(dtype)
    # FIXME: We don't validate the output when this is a string-to-boolean
    #        conversion, as libcudf's semantics is different from Pandas'
    #        (see GH #7875).
    if dtype is not bool:
        assert equals(out_ls, out_s)

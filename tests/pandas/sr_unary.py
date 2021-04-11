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
from numpy.random import randn

from legate import pandas as lp

s = pd.Series(randn(17) * 10)
ls = lp.Series(s)

print("Testing abs")
out_s = s.abs()
out_ls = ls.abs()
assert out_ls.equals(out_s)

print("Testing negation")
out_s = -s
out_ls = -ls
assert out_ls.equals(out_s)

print("Testing bitwise invert")
out_s = ~s.astype(np.int64)
out_ls = ~ls.astype(np.int64)
assert out_ls.equals(out_s)

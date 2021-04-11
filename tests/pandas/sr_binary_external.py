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
from numpy.random import permutation, randn

from legate import pandas as lp
from tests.utils import equals

n = 17
indices = [pd.RangeIndex(1, n + 1), pd.Index(permutation(n))]

for index in indices:
    print(f"Index: {index}")
    s1 = pd.Series(randn(n), index=index)
    ls1 = lp.Series(s1)
    s2 = pd.Series(randn(n), index=index)

    out_s = s1 + s2
    out_ls = ls1 + s2
    assert equals(out_ls, out_s)

    out_s = s1 + s2.values
    out_ls = ls1 + s2.values
    assert equals(out_ls, out_s)

    out_s = s1 + s2.to_list()
    out_ls = ls1 + s2.to_list()
    assert equals(out_ls, out_s)

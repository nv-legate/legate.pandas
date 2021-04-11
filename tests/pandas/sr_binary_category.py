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
from tests.utils import equals

n = 15
categories = ["C", "A", "B", "D", "E"]
indices = np.random.randint(0, 5, n, dtype=np.int64)

cat1_type = pd.CategoricalDtype(categories=list("BCAD"), ordered=False)
cat2_type = pd.CategoricalDtype(categories=list("ABCD"), ordered=False)

s1 = pd.Series([categories[i] for i in indices], dtype=cat1_type)
s2 = pd.Series([categories[i] for i in indices], dtype=cat2_type)

ls1 = lp.Series(s1)
ls2 = lp.Series(s2)

out_s = s1 == s1
out_ls = ls1 == ls1
assert equals(out_ls, out_s)

out_s = s1 == s2
out_ls = ls1 == ls2
assert equals(out_ls, out_s)

out_s = s1 != s1
out_ls = ls1 != ls1
assert equals(out_ls, out_s)

out_s = s1 != s2
out_ls = ls1 != ls2
assert equals(out_ls, out_s)

out_s = s1 == "C"
out_ls = ls1 == "C"
assert equals(out_ls, out_s)

out_s = s1 == "E"
out_ls = ls1 == "E"
assert equals(out_ls, out_s)

out_s = s1 != "B"
out_ls = ls1 != "B"
assert equals(out_ls, out_s)

out_s = s1 != "E"
out_ls = ls1 != "E"
assert equals(out_ls, out_s)

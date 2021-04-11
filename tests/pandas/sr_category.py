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

n = 15
categories = ["C", "A", "B", "D"]
indices = np.random.randint(0, 4, n, dtype=np.int64)

cat_type = pd.CategoricalDtype(categories=list("BCAD"), ordered=True)
s = pd.Series([categories[i] for i in indices], dtype=cat_type)
ls = lp.Series(s)
assert ls.equals(s)

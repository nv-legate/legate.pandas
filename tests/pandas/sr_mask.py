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

for n in [10, 20, 40]:
    df = pd.DataFrame({"c1": list(range(n)), "c2": [1234] * n})
    ldf = lp.DataFrame(df)

    out_s = df["c1"].mask(df["c1"] % 2 == 0, df["c2"], axis=0)
    out_ls = ldf["c1"].mask(ldf["c1"] % 2 == 0, ldf["c2"], axis=0)

    assert out_ls.equals(out_s)

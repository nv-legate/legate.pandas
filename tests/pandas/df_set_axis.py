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

df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
ldf = lp.DataFrame(df)

idx = [100, 200, 300]
out_df = df.set_axis(idx, axis=0)
out_ldf = ldf.set_axis(idx, axis=0)

assert out_ldf.equals(out_df)

idx = ["A", "B", "C"]
out_df = df.set_axis(idx, axis=0)
out_ldf = ldf.set_axis(idx, axis=0)

assert out_ldf.equals(out_df)

out_df = df.set_axis(df["a"], axis=0)
out_ldf = ldf.set_axis(ldf["a"], axis=0)

assert out_ldf.equals(out_df)

out_df = df.set_axis(["A", "B", "C"], axis=1)
out_ldf = ldf.set_axis(["A", "B", "C"], axis=1)

assert out_ldf.equals(out_df)

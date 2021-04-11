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
from tests.utils import equals

df = pd.DataFrame({1: [1, 2, 3], 2: [4, 5, 6]})
sr = pd.Series([7, 8, 9])
out_df = pd.concat([sr, df, sr, sr, df], axis=1)

ldf = lp.DataFrame(df)
lsr = lp.Series(sr)
out_ldf = lp.concat([sr, ldf, lsr, sr, ldf], axis=1)

assert equals(out_ldf, out_df)

out_sr = pd.concat([sr, sr, sr, sr], axis=0)

out_lsr = lp.concat([sr, lsr, lsr, sr], axis=0)

out_sr = out_sr.sort_index()
out_lsr = out_lsr.sort_index()

assert equals(out_lsr, out_sr)

df = pd.DataFrame({1: [1, 2, 3], 2: [4, 5, 6]})
out_df = pd.concat([df, df, df])

ldf = lp.DataFrame(df)
out_ldf = lp.concat([ldf, ldf, ldf])

out_ldf = pd.DataFrame(
    {1: out_ldf[1].to_numpy(), 2: out_ldf[2].to_numpy()}, index=out_ldf.index
)

out_df = out_df.sort_values(by=[1, 2])
out_ldf = out_ldf.sort_values(by=[1, 2])

assert equals(out_ldf, out_df)

out_df = df.append(df)
out_ldf = ldf.append(ldf)

out_df = out_df.sort_index()
out_ldf = out_ldf.sort_index()

assert equals(out_ldf, out_df)

out_sr = sr.append(sr)
out_lsr = lsr.append(lsr)

out_sr = out_sr.sort_index()
out_lsr = out_lsr.sort_index()

assert equals(out_lsr, out_sr)

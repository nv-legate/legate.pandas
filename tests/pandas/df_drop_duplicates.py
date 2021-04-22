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

df = pd.DataFrame(
    {
        "brand": ["Yum Yum", "Yum Yum", "Indomie", "Indomie", "Indomie"],
        "style": ["cup", "cup", "cup", "pack", "pack"],
        "rating": [4, 4, 3.5, 15, 5],
    }
)
ldf = lp.DataFrame(df)

out_df = df.drop_duplicates().sort_index()
out_ldf = ldf.drop_duplicates().sort_index()

assert equals(lp.DataFrame(out_df), out_ldf)

out_df = df.drop_duplicates(subset=["brand"]).sort_index()
out_ldf = ldf.drop_duplicates(subset=["brand"]).sort_index()

assert equals(lp.DataFrame(out_df), out_ldf)

out_df = df.drop_duplicates(
    subset=["brand", "style"], keep="last"
).sort_index()
out_ldf = ldf.drop_duplicates(
    subset=["brand", "style"], keep="last"
).sort_index()

assert equals(out_ldf, out_df)

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

import os
import shutil
import tempfile

import pandas as pd
from numpy.random import permutation

from legate import pandas as lp

n = 10000

for index in [
    pd.Index(permutation(n)),
    pd.Index(permutation(n), name="icol"),
    pd.MultiIndex.from_arrays([permutation(n), permutation(n)]),
    pd.MultiIndex.from_arrays(
        [permutation(n), permutation(n)], names=("icol1", None)
    ),
    pd.RangeIndex(n),
    pd.RangeIndex(3 * n + 1, 1, -3, name="k"),
]:
    df = pd.DataFrame(
        {
            "a": range(n, 0, -1),
            "b": range(n),
            "c": [str(i) * 3 for i in range(n)],
        },
        index=index,
    )
    df["a"] = df["a"].astype("int32")
    df["b"] = df["b"].astype("float64")
    df["c"] = df["c"].astype(pd.StringDtype())

    ldf = lp.DataFrame(df)

    for store_index in [None, False, True]:
        print(f"Index type: {type(index)}, store index?: {store_index}")

        path = os.path.join(os.path.dirname(__file__), "files")
        tmp_dir = tempfile.mkdtemp(dir=path)
        out_path = os.path.sep.join([tmp_dir, "out.parquet"])

        print(f"Dump to {out_path}")

        try:
            ldf.to_parquet(out_path, index=store_index)
            df_copy = pd.read_parquet(out_path)
            ldf_copy = lp.read_parquet(out_path)
            if store_index is not False:
                assert ldf_copy.equals(df)
                assert ldf_copy.equals(df_copy)
            else:
                assert ldf_copy.equals(df.reset_index(drop=True))
                assert ldf_copy.equals(df_copy.reset_index(drop=True))
        finally:
            shutil.rmtree(tmp_dir)

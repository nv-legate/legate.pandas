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

import itertools

import numpy as np
import pandas as pd
from numpy.random import randn

from legate import pandas as lp

key_dtypes = [np.int32, np.int64]
# XXX: Pandas is really weird in its handling of join keys. Specifically,
#      if the index of the left table is UIntXXIndex or FloatXXIndex and
#      it is used for the join, Pandas will always convert it into
#      Int64Index, even though there is no valid reason for doing that.
#      In Legate, we will not do this type conversion, as it's really
#      bad for performance, given that the index can be stored across
#      all the nodes in the system.
# key_dtypes += [np.uint32, np.uint64, np.float32, np.float64]


def to_pandas(ldf):
    columns = {}
    for column in ldf.columns:
        columns[column] = ldf[column].to_numpy()
    return pd.DataFrame(columns)


def sort_and_compare(df1, df2):
    df1 = df1.sort_values(by=df1.columns.to_list(), ignore_index=True)
    df2 = df2.sort_values(by=df1.columns.to_list(), ignore_index=True)
    return df1.equals(df2)


for n in [15, 30, 45]:
    c1 = np.array(randn(n) * 100.0, dtype=np.int64)
    c2 = np.array(randn(n // 3) * 100.0, dtype=np.int64)
    c3_l = np.array(randn(n) * 100.0, dtype=np.int64)
    c3_r = np.array(randn(n // 3) * 100.0, dtype=np.int64)

    key_right = list(range(n // 3))
    key_left = list(itertools.chain(*[[x] * 3 for x in range(n // 3)]))
    for key_dtype in key_dtypes:
        print("Type: inner, Size: %u, Key dtype: %s" % (n, str(key_dtype)))

        df1 = pd.DataFrame(
            {"c1": c1, "key1": np.array(key_left, dtype=key_dtype), "c3": c3_l}
        )
        df1["key"] = df1["key1"]
        df1_key_on_index = df1.set_index("key")

        df2 = pd.DataFrame(
            {
                "c2": c2,
                "key2": np.array(key_right, dtype=key_dtype),
                "c3": c3_r,
            }
        )
        df2["key"] = df2["key2"]
        df2_key_on_index = df2.set_index("key")

        ldf1 = lp.DataFrame(df1)
        ldf1_key_on_index = lp.DataFrame(df1_key_on_index)
        ldf2 = lp.DataFrame(df2)
        ldf2_key_on_index = lp.DataFrame(df2_key_on_index)

        join_pandas2 = df1.merge(
            df2_key_on_index, left_on="key1", right_index=True
        )
        join_pandas4 = df1_key_on_index.merge(
            df2, right_on="key2", left_index=True
        )
        # XXX: Pandas sort the keys in the output when both left_index and
        #      right_index are True, whereas Legate will not for performance
        #      reasons. In this test we sorted the keys in the input dataframe
        #      so that Pandas' join output coincides with Legate's. We can't
        #      and won't guarantee this semantics equivalence in general.
        join_pandas5 = df1_key_on_index.merge(
            df2_key_on_index, left_index=True, right_index=True
        )
        join_pandas6 = df1.merge(df2, left_on="key1", right_on="key2")

        join_legate2 = ldf1.merge(
            ldf2_key_on_index,
            left_on="key1",
            right_index=True,
            method="broadcast",
        )
        assert sort_and_compare(join_pandas2, to_pandas(join_legate2))
        join_legate4 = ldf1_key_on_index.merge(
            ldf2, right_on="key2", left_index=True, method="broadcast"
        )
        assert sort_and_compare(join_pandas4, to_pandas(join_legate4))
        join_legate5 = ldf1_key_on_index.merge(
            ldf2_key_on_index,
            left_index=True,
            right_index=True,
            method="broadcast",
        )
        assert sort_and_compare(join_pandas5, to_pandas(join_legate5))
        join_legate6 = ldf1.merge(
            ldf2, left_on="key1", right_on="key2", method="broadcast"
        )
        assert sort_and_compare(join_pandas6, to_pandas(join_legate6))

        join_legate_hash2 = ldf1.merge(
            ldf2_key_on_index, left_on="key1", right_index=True, method="hash"
        )
        assert sort_and_compare(join_pandas2, to_pandas(join_legate_hash2))
        join_legate_hash4 = ldf1_key_on_index.merge(
            ldf2, right_on="key2", left_index=True, method="hash"
        )
        assert sort_and_compare(join_pandas4, to_pandas(join_legate_hash4))
        join_legate_hash5 = ldf1_key_on_index.merge(
            ldf2_key_on_index, left_index=True, right_index=True, method="hash"
        )
        assert sort_and_compare(join_pandas5, to_pandas(join_legate_hash5))
        join_legate_hash6 = ldf1.merge(
            ldf2, left_on="key1", right_on="key2", method="hash"
        )
        assert sort_and_compare(join_pandas6, to_pandas(join_legate_hash6))

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

# key_dtypes = [np.int32, np.int64, np.float32, np.float64]
key_dtypes = [np.int32]


def to_pandas(ldf):
    columns = {}
    for column in ldf.columns:
        columns[column] = ldf[column].to_numpy()
    return pd.DataFrame(columns)


def sort_and_compare(df1, df2):
    df1 = df1.sort_values(by=df1.columns.to_list(), ignore_index=True)
    df2 = df2.sort_values(by=df1.columns.to_list(), ignore_index=True)
    print(df1)
    print(df2)
    return df1.equals(df2)


all_columns = ["key", "payload_x", "payload_y"]
for n in [15, 30, 45]:
    c_l = np.array(randn(n) * 100.0, dtype=np.int64)
    c_r = np.array(randn(n // 3) * 100.0, dtype=np.int64)

    key_right = list(range(n // 3))
    key_left = list(
        itertools.chain(*[[x] * 3 for x in range(n // 3 - 1, -1, -1)])
    )
    for key_dtype in key_dtypes:
        print("Type: inner, Size: %u, Key dtype: %s" % (n, str(key_dtype)))

        df1 = pd.DataFrame(
            {"key": np.array(key_left, dtype=key_dtype), "payload": c_l}
        )
        df2 = pd.DataFrame(
            {"key": np.array(key_right, dtype=key_dtype), "payload": c_r}
        )
        df1["payload"] = df1["payload"].astype(pd.StringDtype()).str.zfill(10)
        df2["payload"] = df2["payload"].astype(pd.StringDtype()).str.zfill(10)

        ldf1 = lp.DataFrame(df1)
        ldf2 = lp.DataFrame(df2)
        ldf1["payload"] = (
            ldf1["payload"].astype(pd.StringDtype()).str.zfill(10)
        )
        ldf2["payload"] = (
            ldf2["payload"].astype(pd.StringDtype()).str.zfill(10)
        )

        join_pandas = df1.merge(df2, on="key")
        join_legate = ldf1.merge(ldf2, on="key", method="broadcast")
        join_legate_hash = ldf1.merge(ldf2, on="key", method="hash")

        assert join_legate.sort_values(
            by=all_columns, ignore_index=True
        ).equals(join_pandas.sort_values(by=all_columns, ignore_index=True))
        assert join_legate_hash.sort_values(
            by=all_columns, ignore_index=True
        ).equals(join_pandas.sort_values(by=all_columns, ignore_index=True))

    key_left = list(itertools.chain(*[[x] * 3 for x in range(n // 3, 0, -1)]))
    for key_dtype in key_dtypes:
        # print("Type: left, Size: %u, Key dtype: %s" % (n, str(key_dtype)))

        df1 = pd.DataFrame(
            {"key": np.array(key_left, dtype=key_dtype), "payload": c_l}
        )
        df2 = pd.DataFrame(
            {"key": np.array(key_right, dtype=key_dtype), "payload": c_r}
        )
        df1["payload"] = df1["payload"].astype(pd.StringDtype()).str.zfill(10)
        df2["payload"] = df2["payload"].astype(pd.StringDtype()).str.zfill(10)

        ldf1 = lp.DataFrame(df1)
        ldf2 = lp.DataFrame(df2)
        ldf1["payload"] = (
            ldf1["payload"].astype(pd.StringDtype()).str.zfill(10)
        )
        ldf2["payload"] = (
            ldf2["payload"].astype(pd.StringDtype()).str.zfill(10)
        )

        join_pandas = df1.merge(df2, on="key", how="left")
        join_legate = ldf1.merge(
            ldf2, on="key", how="left", method="broadcast"
        )
        join_legate_hash = ldf1.merge(
            ldf2, on="key", how="left", method="hash"
        )

        assert join_legate.sort_values(
            by=all_columns, ignore_index=True
        ).equals(join_pandas.sort_values(by=all_columns, ignore_index=True))
        assert join_legate_hash.sort_values(
            by=all_columns, ignore_index=True
        ).equals(join_pandas.sort_values(by=all_columns, ignore_index=True))

        print("Type: outer, Size: %u, Key dtype: %s" % (n, str(key_dtype)))

        df1["key"] = df1["key"].astype(pd.StringDtype()).str.zfill(3)
        df2["key"] = df2["key"].astype(pd.StringDtype()).str.zfill(3)

        ldf1["key"] = ldf1["key"].astype(pd.StringDtype()).str.zfill(3)
        ldf2["key"] = ldf2["key"].astype(pd.StringDtype()).str.zfill(3)

        join_pandas = df1.merge(df2, on="key", how="outer")

        join_legate_hash = ldf1.merge(
            ldf2, on="key", how="outer", method="hash"
        )

        sorted_pandas = join_pandas.sort_values(
            by=["payload_x", "payload_y", "key"], ignore_index=True
        )
        sorted_legate = join_legate_hash.sort_values(
            by=["payload_x", "payload_y", "key"], ignore_index=True
        )

        assert sorted_legate.equals(sorted_pandas)

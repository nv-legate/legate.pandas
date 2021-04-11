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

import pandas
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def similar_series(a, b):
    if not is_numeric_dtype(a.dtype) or is_bool_dtype(a.dtype):
        return False
    elif not a.index.equals(b.index):
        return False
    return ((a.fillna(0) - b.fillna(0)).abs() < 1e-10).all()


def similar(a, b):
    if isinstance(b, pandas.Series):
        return similar_series(a, b)
    if not a.columns.equals(b.columns):
        return False
    for c in a.columns:
        if not (a[c].equals(b[c]) or similar_series(a[c], b[c])):
            return False
    return True


def equals(df1, df2, use_RHS_columns_only=False):
    if not use_RHS_columns_only:
        result = df1.equals(df2) or similar(df1, df2)
    else:
        result = True
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)
        for c in df2.columns:
            if not (df1[c].equals(df2[c]) or similar_series(df1[c], df2[c])):
                result = False
                break

    if not result:
        print(df1)
        print(df2)
    return result


def equals_scalar(out_lp, out_pd):
    result = out_lp == out_pd
    if not result:
        print(out_lp, out_pd)
    return result


def must_fail(ex, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        raise RuntimeError("test failed")
    except ex as err:
        print(f"Catched {ex.__name__}: {err}")
        return
    raise RuntimeError("test failed")

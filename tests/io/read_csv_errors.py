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

from legate import pandas as lp
from tests.utils import must_fail

path = os.path.join(os.path.dirname(__file__), "files", "read_csv.csv")


def _test(ex, paths, **kwargs):
    must_fail(ex, lp.read_csv, paths, **kwargs)


_test(NotImplementedError, 1)
_test(ValueError, [])
_test(ValueError, path, sep=None, delimiter=None)
_test(ValueError, path, sep="ab")
_test(NotImplementedError, path, header=1)
_test(ValueError, path, skiprows="a")
_test(ValueError, path, skipfooter="a")
_test(ValueError, path, nrows="a")
_test(NotImplementedError, path, names=["a", "b"], dtype=[])
_test(ValueError, path, names=["a", "b"], dtype={"a": "float64"})
_test(ValueError, path, names=["a", "b"], dtype={"a": "float64", "b": "flt"})
_test(NotImplementedError, path, prefix="X")
_test(NotImplementedError, path, mangle_dupe_cols=False)
_test(NotImplementedError, path, parse_dates={"a": [0, 1]})
_test(
    NotImplementedError,
    path,
    names=["a", "b"],
    dtype={"a": "float64", "b": "float64"},
    parse_dates=0,
)
_test(
    KeyError,
    path,
    names=["a", "b"],
    dtype={"a": "float64", "b": "float64"},
    parse_dates=["c"],
)
_test(NotImplementedError, path, quoting=1)
_test(
    ValueError,
    path,
    names=["a", "b"],
    dtype={"a": "float64", "b": "float64"},
    quotechar="ab",
)
_test(
    NotImplementedError,
    path,
    names=["a", "b"],
    dtype={"a": "float64", "b": "float64"},
    index_col=1.5,
)
_test(
    KeyError,
    path,
    names=["a", "b"],
    dtype={"a": "float64", "b": "float64"},
    index_col="c",
)
_test(
    ValueError,
    path,
    names=["a", "b"],
    dtype={"a": "float64", "b": "float64"},
    true_values=[1, 2],
)
_test(
    ValueError,
    path,
    names=["a", "b"],
    dtype={"a": "float64", "b": "float64"},
    false_values=1,
)
_test(
    ValueError,
    path,
    names=["a", "b"],
    dtype={"a": "float64", "b": "float64"},
    na_values=1.0,
)
_test(
    ValueError,
    path,
    names=["a", "b"],
    dtype={"a": "float64", "b": "float64"},
    skipfooter=1,
    nrows=1,
)

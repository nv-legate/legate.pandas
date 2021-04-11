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
from tests.utils import equals, must_fail


def _test(ex, df, *args, **kwargs):
    must_fail(ex, df.drop, *args, **kwargs)


df = pd.DataFrame(
    {
        (1, "a", "A"): [1, 1, 1, 2, 3, 3],
        (2, "b", "B"): [1, 2, 3, 1, 2, 3],
        (3, "c", "C"): [1, 1, 1, 2, 2, 2],
    },
    index=pd.MultiIndex.from_arrays(
        [[1, 1, 2, 2, 3, 3], [1, 2, 3, 4, 5, 6], [5, 4, 3, 2, 1, 0]]
    ).rename(("idx_1", "idx_2", "idx_3")),
)
df.columns.names = ("col_1", "col_2", "col_3")
ldf = lp.DataFrame(df)

assert equals(ldf.drop(1, axis=1), df.drop(1, axis=1))
assert equals(ldf.drop("b", axis=1, level=1), df.drop("b", axis=1, level=1))
assert equals(
    ldf.drop("C", axis=1, level="col_3"), df.drop("C", axis=1, level="col_3")
)
assert equals(ldf.drop((3, "c"), axis=1), df.drop((3, "c"), axis=1))
assert equals(ldf.drop((1, "a", "A"), axis=1), df.drop((1, "a", "A"), axis=1))

_test(ValueError, ldf, labels="C", columns="C")
_test(KeyError, ldf, "D", axis=1, level=2)
_test(KeyError, ldf, (1, 2, 3, 4), axis=1, level=2)

assert equals(ldf.drop((1, 2, 4), axis=0), df.drop((1, 2, 4), axis=0))
assert equals(ldf.drop((2, 3), axis=0), df.drop((2, 3), axis=0))
assert equals(ldf.drop(3, axis=0, level=0), df.drop(3, axis=0, level=0))
assert equals(
    ldf.drop(3, axis=0, level="idx_2"), df.drop(3, axis=0, level="idx_2")
)

assert equals(
    ldf.drop(index=3, columns=(2, "b")), df.drop(index=3, columns=(2, "b"))
)
assert equals(
    ldf.drop(index=3, columns=[2, 3], level=0),
    df.drop(index=3, columns=[2, 3], level=0),
)

sr = df[(1, "a", "A")]
lsr = ldf[(1, "a", "A")]

assert equals(lsr.drop((1, 2, 4), axis=0), sr.drop((1, 2, 4), axis=0))
assert equals(lsr.drop((2, 3), axis=0), sr.drop((2, 3), axis=0))
assert equals(lsr.drop(3, axis=0, level=0), sr.drop(3, axis=0, level=0))

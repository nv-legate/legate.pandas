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

try:
    from legate.numpy.random import random
except ModuleNotFoundError:
    from numpy.random import random

import pandas as pd

from legate import pandas as lp

x = random(10)
y = random(10)
df = pd.DataFrame({"x": x, "y": y})

ldf1 = lp.DataFrame({"x": x, "y": y})
# FIXME: We don't handle this case correctly now. DataFrame's ctor
#        should align all series in the dictionary.
# ldf2 = lp.DataFrame({"x": lp.Series(x), "y": lp.Series(y)})
ldf3 = lp.DataFrame(ldf1)

assert ldf1.equals(lp.DataFrame(df))
# assert ldf2.equals(lp.DataFrame(df))
assert ldf3.equals(lp.DataFrame(df))

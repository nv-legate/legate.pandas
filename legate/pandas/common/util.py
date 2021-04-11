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

from pandas import Timestamp
from pandas.core.dtypes.common import is_list_like


def to_list_if_scalar(val):
    if not is_list_like(val) or isinstance(val, tuple):
        return [val]
    else:
        return list(val)


def to_list_if_not_none(v):
    if v is None:
        return v
    return to_list_if_scalar(v)


def either(a, b):
    return a if a is not None else b


def get_dtypes(columns):
    return [column.dtype for column in columns]


def ifilter(f, ls):
    return [v for i, v in filter(lambda pair: f(pair[0]), enumerate(ls))]


def ith_list(ls, i):
    return [t[i] for t in ls]


def fst_list(ls):
    return [fst for (fst, snd) in ls]


def snd_list(ls):
    return [snd for (fst, snd) in ls]


def ith_set(ls, i):
    return set(t[i] for t in ls)


def fst_set(ls):
    return set(fst for (fst, snd) in ls)


def snd_set(ls):
    return set(snd for (fst, snd) in ls)


def is_tuple(v):
    return isinstance(v, tuple)


def sanitize_scalar(scalar):
    if isinstance(scalar, Timestamp):
        return scalar.to_numpy()

    else:
        return scalar


def to_tuple_if_scalar(val):
    return val if is_tuple(val) else (val,)

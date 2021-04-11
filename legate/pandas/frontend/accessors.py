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

import numpy as np

from legate.pandas.common import types as ty


class SeriesAccessor(object):
    def __init__(self, series):
        self._series = series
        self._column = series._frame._columns[0]

    def _construct_result(self, column):
        return self._series.__ctor__(
            frame=self._series._frame.replace_columns([column])
        )


class CategoricalAccessor(SeriesAccessor):
    def __init__(self, series):
        super(CategoricalAccessor, self).__init__(series)
        assert ty.is_categorical_dtype(self._column.dtype)

    @property
    def codes(self):
        return self._construct_result(self._column.get_codes())


class DatetimeProperties(SeriesAccessor):
    def __init__(self, series):
        super(DatetimeProperties, self).__init__(series)
        assert self._column.dtype == ty.ts_ns

    @property
    def year(self):
        return self._get_dt_field("year")

    @property
    def month(self):
        return self._get_dt_field("month")

    @property
    def day(self):
        return self._get_dt_field("day")

    @property
    def hour(self):
        return self._get_dt_field("hour")

    @property
    def minute(self):
        return self._get_dt_field("minute")

    @property
    def second(self):
        return self._get_dt_field("second")

    @property
    def weekday(self):
        return self._get_dt_field("weekday")

    def _get_dt_field(self, field):
        dtype = ty.get_dt_field_type(self._column.dtype, field)
        return self._construct_result(self._column.get_dt_field(field, dtype))


class StringMethods(SeriesAccessor):
    def __init__(self, series):
        super(StringMethods, self).__init__(series)
        assert ty.is_string_dtype(self._column.dtype)

    def contains(self, pat, case=True, flags=0, na=np.NaN, regex=True):
        if pat is None and not case:
            raise AttributeError("'NoneType' object has no attribute 'upper'")
        assert pat is not None and regex
        return self._construct_result(self._column.contains(pat))

    def pad(self, width, side="left", fillchar=" "):
        if len(fillchar) != 1:
            raise TypeError("fillchar must be a character, not str")
        return self._construct_result(
            self._column.pad(width, side=side, fillchar=fillchar)
        )

    def strip(self, to_strip=None):
        return self._construct_result(self._column.strip(to_strip=to_strip))

    def zfill(self, width):
        return self._construct_result(self._column.zfill(width))

    def lower(self):
        return self._construct_result(self._column.lower())

    def upper(self):
        return self._construct_result(self._column.upper())

    def swapcase(self):
        return self._construct_result(self._column.swapcase())

    def to_datetime(self, format):
        if format is None:
            raise ValueError("Format must be provided")

        return self._construct_result(self._column.to_datetime(format))

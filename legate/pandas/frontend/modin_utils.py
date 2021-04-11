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

# Licensed to Modin Development Team under one or more contributor license
# agreements. See the MODIN_NOTICE file distributed with this work for
# additional information regarding copyright ownership. The Modin Development
# Team licenses this file to you under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with the
# License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

import pandas
from pandas.io.formats import console

#########################################################################
# The following functions are originally from Modin and adapted to Legate
# Pandas.
#########################################################################


def _repr_frame(frame, columns, num_rows, num_cols):
    num_rows_for_head = num_rows // 1 + 1
    num_cols_for_front = num_cols // 2 + 1

    if len(frame) <= num_rows:
        head = frame._frame
        tail = None
    else:
        head = frame._frame.head(num_rows_for_head)
        tail = frame._frame.tail(num_rows_for_head)

    if frame._is_series or len(columns) <= num_cols:
        head_front = head.to_pandas()
        head_back = pandas.DataFrame()
        tail_back = pandas.DataFrame()

        if tail is not None:
            tail_front = tail.to_pandas()
        else:
            tail_front = pandas.DataFrame(columns=head_front.columns)

    else:
        columns = columns[:num_cols_for_front].append(
            columns[-num_cols_for_front:]
        )
        head_front = head.front(num_cols_for_front).to_pandas()

        head_back = head.back(num_cols_for_front).to_pandas()
        head_back = head_back.rename(
            columns=lambda idx: idx + num_cols_for_front
        )

        if tail is not None:
            tail_front = tail.front(num_cols_for_front).to_pandas()
            tail_back = tail.back(num_cols_for_front).to_pandas()
            tail_back = tail_back.rename(
                columns=lambda idx: idx + num_cols_for_front
            )
        else:
            tail_front = tail_back = pandas.DataFrame()

    head_for_repr = pandas.concat([head_front, head_back], axis=1)
    tail_for_repr = pandas.concat([tail_front, tail_back], axis=1)

    df = pandas.concat([head_for_repr, tail_for_repr])
    df.columns = columns
    return df


def _repr_dataframe(df):
    num_rows = pandas.get_option("display.max_rows") or 10
    num_cols = pandas.get_option("display.max_columns") or 20

    if pandas.get_option("display.max_columns") is None and pandas.get_option(
        "display.expand_frame_repr"
    ):
        width, _ = console.get_console_size()
        col_counter = 0
        i = 0
        while col_counter < width:
            col_counter += len(str(df.columns[i])) + 1
            i += 1

        num_cols = i
        i = len(df.columns) - 1
        col_counter = 0
        while col_counter < width:
            col_counter += len(str(df.columns[i])) + 1
            i -= 1

        num_cols += len(df.columns) - i
    result = repr(_repr_frame(df, df.columns, num_rows, num_cols))
    if len(df) > num_rows or len(df.columns) > num_cols:
        return (
            result.rsplit("\n\n", 1)[0]
            + f"\n\n[{len(df)} rows x {len(df.columns)} columns]"
        )
    else:
        return result


def _repr_series(sr):
    num_rows = pandas.get_option("max_rows") or 60
    num_cols = pandas.get_option("max_columns") or 20
    temp_df = _repr_frame(sr, [sr.name], num_rows, num_cols).squeeze(axis=1)
    temp_str = repr(temp_df)
    if sr.name is not None:
        name_str = f"Name: {sr.name}, "
    else:
        name_str = ""
    if len(sr) > num_rows:
        len_str = f"Length: {len(sr)}, "
    else:
        len_str = ""
    dtype_str = f'dtype: {temp_str.rsplit("dtype: ", 1)[-1]}'
    if len(sr) == 0:
        return f"Series([], {name_str}{dtype_str}"

    if sr.name is not None:
        preamble = temp_str.rsplit("\nName:", 1)[0]
        return preamble + f"\n{name_str}{len_str}{dtype_str}"
    elif len(sr) > num_rows:
        preamble = temp_str.rsplit("\nLength:", 1)[0]
        return preamble + f"\n{len_str}{dtype_str}"
    else:
        preamble = temp_str.rsplit("\ndtype:", 1)[0]
        return preamble + f"\n{dtype_str}"

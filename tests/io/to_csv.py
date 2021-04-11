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

import glob
import itertools
import os
import shutil
import tempfile

import pandas as pd
from numpy.random import permutation

from legate import pandas as lp

n = 100

data = {
    "col_a": range(n, 0, -1),
    "col_b": range(n),
    "col_c": ["prefix" + str(i) * 3 for i in range(n)],
}

dtype = {"col_a": "int32", "col_b": "float64", "col_c": "string"}

column_names = list(data.keys())

all_params = {
    "params_header": [True, False],
    "params_index": [True, False],
    "params_partition": [True, False],
    "params_columns": [None, [column_names[0], column_names[2]]],
    "params_sep": [",", "|"],
}


def sanitize_column_names(names):
    return column_names if names is None else names


def read_csv(fn, path, conf, index):
    sep = conf["sep"]

    if conf["index"]:
        if conf["header"]:
            return fn(
                path,
                sep=sep,
                dtype=dtype,
                index_col=list(range(index.nlevels)),
            )
        else:
            index_names = [f"ind_{lvl}" for lvl in range(index.nlevels)]
            names = index_names + sanitize_column_names(conf["columns"])
            for index_name in index_names:
                dtype[index_name] = "int64"
            return fn(
                path,
                sep=sep,
                names=names,
                dtype=dtype,
                index_col=names[: index.nlevels],
            )
    else:
        if conf["header"]:
            return fn(path, sep=sep, dtype=dtype)
        else:
            names = sanitize_column_names(conf["columns"])
            return fn(path, sep=sep, names=names, dtype=dtype)


for index in [
    pd.Index(permutation(n)),
    pd.MultiIndex.from_arrays(
        [permutation(n), permutation(n)], names=("icol1", "icol2")
    ),
    pd.RangeIndex(3 * n + 1, 1, -3, name="k"),
]:
    df = pd.DataFrame(data, index=index)
    for column in column_names:
        df[column] = df[column].astype(dtype[column])

    ldf = lp.DataFrame(df)

    for tpl in itertools.product(*all_params.values()):
        conf = {
            "header": tpl[0],
            "index": tpl[1],
            "partition": tpl[2],
            "columns": tpl[3],
            "sep": tpl[4],
        }

        print(f"Index type: {type(index)}, Conf: {conf}")

        path = os.path.join(os.path.dirname(__file__), "files")
        tmp_dir = tempfile.mkdtemp(dir=path)
        out_path = os.path.sep.join([tmp_dir, "out.csv"])

        print(f"Dump to {out_path}")

        try:
            ldf.to_csv(out_path, **conf)
            if conf["partition"]:
                partitions = sorted(glob.glob(out_path + ".*"))
                dfs = [
                    read_csv(pd.read_csv, part, conf, index)
                    for part in partitions
                ]
                df_copy = pd.concat(dfs)
            else:
                df_copy = read_csv(pd.read_csv, out_path, conf, index)

            columns = sanitize_column_names(conf["columns"])
            if conf["index"]:
                assert ldf[columns].equals(df_copy)
            else:
                assert (
                    ldf[columns]
                    .reset_index(drop=True)
                    .equals(df_copy.reset_index(drop=True))
                )
        finally:
            shutil.rmtree(tmp_dir)

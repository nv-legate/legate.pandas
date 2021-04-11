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

import argparse


def test(
    size_per_proc=1000,
    num_procs=1,
    num_runs=1,
    ty="int64",
    key_length=10,
    scale_lhs_only=False,
    package="legate",
):
    if package == "legate":
        from legate import numpy as np, pandas as pd
        from legate.numpy.random import randn

    elif package == "cudf":
        import cudf as pd
        import cupy as np
        from cupy.random import randn

    elif package == "pandas":
        import numpy as np
        import pandas as pd
        from numpy.random import randn

    elif package == "dask" or package == "daskcudf":
        import dask.array as da
        import dask.dataframe as df
        import numpy as np

        if package == "daskcudf":
            import cudf

    else:
        print("Unknown dataframe package: %s" % package)
        assert False

    if package == "legate":
        from legate.timing import time

        def block(*args):
            pass

        def get_timestamp():
            return time()

        def compute_elapsed_time(start_ts, stop_ts):
            return (stop_ts - start_ts) / 1000.0

    elif package == "dask" or package == "daskcudf":
        import time

        def block(*args):
            for arg in args:
                arg.compute()

        get_timestamp = time.process_time

        def compute_elapsed_time(start_ts, stop_ts):
            return (stop_ts - start_ts) * 1000.0

    else:
        import time

        def block(*args):
            pass

        get_timestamp = time.process_time

        def compute_elapsed_time(start_ts, stop_ts):
            return (stop_ts - start_ts) * 1000.0

    if scale_lhs_only:
        size = size_per_proc * num_procs
        size_rhs = size // 3

        if package == "dask" or package == "daskcudf":
            # Dask array does not have randn so use arrange
            c1 = da.arange(size, dtype=np.float64, chunks=size_per_proc)
            c2 = da.arange(
                size_rhs,
                dtype=np.float64,
                chunks=(size_per_proc + num_procs - 1) // num_procs,
            )
        else:
            c1 = randn(size)
            c2 = randn(size_rhs)

        key_dtype = np.int64
        if package == "dask" or package == "daskcudf":
            key_left = (
                da.arange(size, dtype=key_dtype, chunks=size_per_proc)
                % size_per_proc
            )
            key_right = da.arange(
                size_rhs,
                dtype=key_dtype,
                chunks=(size_per_proc + num_procs - 1) // num_procs,
            )
            da.multiply(key_right, 3, out=key_right)
        else:
            key_left = np.arange(size, dtype=key_dtype) % size_per_proc
            key_right = np.arange(size_rhs, dtype=key_dtype)
            np.multiply(key_right, 3, out=key_right)

    else:
        size = size_per_proc * num_procs
        size_rhs = size

        if package == "dask" or package == "daskcudf":
            # Dask array does not have randn so use arrange
            c1 = da.arange(size, dtype=np.float64, chunks=size_per_proc)
            c2 = da.arange(size, dtype=np.float64, chunks=size_per_proc)
        else:
            c1 = randn(size)
            c2 = randn(size)

        key_dtype = np.int64
        if package == "dask" or package == "daskcudf":
            key_left = da.arange(size, dtype=key_dtype, chunks=size_per_proc)
            key_right = da.arange(size, dtype=key_dtype, chunks=size_per_proc)
        else:
            key_left = np.arange(size, dtype=key_dtype)
            key_right = np.arange(size, dtype=key_dtype)
        # np.floor_divide(key_right, 3, out=key_right)
        # np.multiply(key_right, 3, out=key_right)

    if package == "dask" or package == "daskcudf":
        df1 = df.multi.concat(
            [df.from_dask_array(a) for a in [c1, key_left]], axis=1
        )
        df1.columns = ["c1", "key"]
        df2 = df.multi.concat(
            [df.from_dask_array(a) for a in [c2, key_right]], axis=1
        )
        df2.columns = ["c2", "key"]
        if package == "daskcudf":
            df1 = df1.map_partitions(cudf.from_pandas)
            df2 = df2.map_partitions(cudf.from_pandas)
    else:
        df1 = pd.DataFrame({"c1": c1, "key": key_left})
        df2 = pd.DataFrame({"c2": c2, "key": key_right})
    df2["key"] = df2["key"] // 3 * 3

    if ty == "string":
        df1["key"] = (
            df1["key"]
            .astype("string")
            .str.pad(width=key_length, side="both", fillchar="0")
        )
        df2["key"] = (
            df2["key"]
            .astype("string")
            .str.pad(width=key_length, side="both", fillchar="0")
        )

    print(
        "Type: inner, Size: %u x %u, Key dtype: %s"
        % (size, size_rhs, str(key_dtype))
    )

    block(df1, df2)

    for i in range(num_runs):
        start_ts = get_timestamp()

        df_result = df1.merge(df2, on="key")

        block(df_result)

        stop_ts = get_timestamp()

        print(
            "[Run %d] Elapsed time: %lf ms"
            % (i + 1, compute_elapsed_time(start_ts, stop_ts))
        )

        del df_result


def driver():
    parser = argparse.ArgumentParser(description="Join micro-benchmark")

    parser.add_argument(
        "--size_per_proc",
        dest="size_per_proc",
        type=int,
        default=1000,
        help="Join table size per processor",
    )

    parser.add_argument(
        "--num_procs",
        dest="num_procs",
        type=int,
        default=1,
        help="Number of processors",
    )

    parser.add_argument(
        "--num_runs",
        dest="num_runs",
        type=int,
        default=1,
        help="Number of runs",
    )

    parser.add_argument(
        "--type",
        dest="ty",
        type=str,
        default="int64",
        help="Data type for merge keys",
    )

    parser.add_argument(
        "--key_length",
        dest="key_length",
        type=int,
        default=10,
        help="Length of string keys",
    )

    parser.add_argument(
        "--scale_lhs_only",
        dest="scale_lhs_only",
        action="store_true",
        required=False,
        default=False,
        help="Scaling only the LHS table",
    )

    parser.add_argument(
        "--package",
        dest="package",
        type=str,
        default="legate",
        help="Dataframe package to use",
    )

    args = parser.parse_args()

    test(**vars(args))


if __name__ == "__main__":
    driver()

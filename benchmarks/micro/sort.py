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
    scale_lhs_only=False,
    package="legate",
    ty="int64",
    key_length=40,
    pad_side="right",
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

    else:
        print("Unknown dataframe package: %s" % package)
        assert False

    if package == "legate":
        from legate.timing import time

        def block():
            pass

        def get_timestamp():
            return time()

        def compute_elapsed_time(start_ts, stop_ts):
            return (stop_ts - start_ts) / 1000.0

    else:
        import time

        def block():
            pass

        get_timestamp = time.process_time

        def compute_elapsed_time(start_ts, stop_ts):
            return (stop_ts - start_ts) * 1000.0

    size = size_per_proc * num_procs

    key = np.arange(size, dtype=np.int64) % size_per_proc
    payload = randn(size)

    df = pd.DataFrame({"key": key, "payload": payload})
    if ty == "int64":
        df["key"] = df["key"] * -1
        ascending = True
    if ty == "string":
        df["key"] = (
            df["key"]
            .astype(str)
            .str.pad(width=key_length, side=pad_side, fillchar="0")
        )
        ascending = False

    print("Size: %u, Key dtype: %s" % (size, df["key"].dtype))

    block()

    for i in range(num_runs):
        start_ts = get_timestamp()

        result = df.sort_values("key", ignore_index=True, ascending=ascending)

        stop_ts = get_timestamp()

        print(
            "[Run %d] Elapsed time: %lf ms"
            % (i + 1, compute_elapsed_time(start_ts, stop_ts))
        )

        del result


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
        help="Number or processors",
    )

    parser.add_argument(
        "--num_runs",
        dest="num_runs",
        type=int,
        default=1,
        help="Number of runs",
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

    parser.add_argument(
        "--type",
        dest="ty",
        type=str,
        default="int64",
        help="Data type for sorting keys",
    )

    parser.add_argument(
        "--key_length",
        dest="key_length",
        type=int,
        default=40,
        help="Length of string keys",
    )

    parser.add_argument(
        "--pad_side",
        dest="pad_side",
        type=str,
        default="right",
        help="Padding side for the sorting keys when they are string",
    )

    args = parser.parse_args()

    test(**vars(args))


if __name__ == "__main__":
    driver()

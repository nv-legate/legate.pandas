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
import glob
import os
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd

# This is a Legate port of the mortgage data example
# (https://github.com/rapidsai/notebooks/blob/branch-0.6/mortgage/E2E.ipynb)
#
# The dataset can be downloaded from this website (use the "1GB splits" one):
#   https://docs.rapids.ai/datasets/mortgage-data

default_ts = (
    np.dtype("datetime64[ms]").type("1970-01-01").astype("datetime64[ms]")
)


def null_workaround(df):
    for column, data_type in df.dtypes.items():
        if str(data_type) == "category":
            df[column] = df[column].cat.codes.astype("int32").fillna(-1)
        if str(data_type) in [
            "int8",
            "int16",
            "int32",
            "int64",
            "float32",
            "float64",
        ]:
            df[column] = df[column].fillna(-1)
    return df


def load_performance_csv(lib, performance_path):
    """Loads performance data

    Returns
    -------
    Legate DataFrame
    """

    cols = [
        "loan_id",
        "monthly_reporting_period",
        "servicer",
        "interest_rate",
        "current_actual_upb",
        "loan_age",
        "remaining_months_to_legal_maturity",
        "adj_remaining_months_to_maturity",
        "maturity_date",
        "msa",
        "current_loan_delinquency_status",
        "mod_flag",
        "zero_balance_code",
        "zero_balance_effective_date",
        "last_paid_installment_date",
        "foreclosed_after",
        "disposition_date",
        "foreclosure_costs",
        "prop_preservation_and_repair_costs",
        "asset_recovery_costs",
        "misc_holding_expenses",
        "holding_taxes",
        "net_sale_proceeds",
        "credit_enhancement_proceeds",
        "repurchase_make_whole_proceeds",
        "other_foreclosure_proceeds",
        "non_interest_bearing_upb",
        "principal_forgiveness_upb",
        "repurchase_make_whole_proceeds_flag",
        "foreclosure_principal_write_off_amount",
        "servicing_activity_indicator",
    ]

    dtypes = OrderedDict(
        [
            ("loan_id", "int64"),
            ("monthly_reporting_period", "date"),
            ("servicer", "category"),
            ("interest_rate", "float64"),
            ("current_actual_upb", "float64"),
            ("loan_age", "float64"),
            ("remaining_months_to_legal_maturity", "float64"),
            ("adj_remaining_months_to_maturity", "float64"),
            ("maturity_date", "date"),
            ("msa", "float64"),
            ("current_loan_delinquency_status", "int32"),
            ("mod_flag", "category"),
            ("zero_balance_code", "category"),
            ("zero_balance_effective_date", "date"),
            ("last_paid_installment_date", "date"),
            ("foreclosed_after", "date"),
            ("disposition_date", "date"),
            ("foreclosure_costs", "float64"),
            ("prop_preservation_and_repair_costs", "float64"),
            ("asset_recovery_costs", "float64"),
            ("misc_holding_expenses", "float64"),
            ("holding_taxes", "float64"),
            ("net_sale_proceeds", "float64"),
            ("credit_enhancement_proceeds", "float64"),
            ("repurchase_make_whole_proceeds", "float64"),
            ("other_foreclosure_proceeds", "float64"),
            ("non_interest_bearing_upb", "float64"),
            ("principal_forgiveness_upb", "float64"),
            ("repurchase_make_whole_proceeds_flag", "category"),
            ("foreclosure_principal_write_off_amount", "float64"),
            ("servicing_activity_indicator", "category"),
        ]
    )

    def dateparse(x):
        if not isinstance(x, str):
            return pd.NaT
        try:
            return datetime.strptime(x, "%m/%d/%Y")
        except ValueError:
            return datetime.strptime(x, "%m/%Y")

    df = lib.read_csv(
        performance_path,
        names=cols,
        delimiter="|",
        dtype=dtypes,
        parse_dates=[
            "monthly_reporting_period",
            "maturity_date",
            "zero_balance_effective_date",
            "last_paid_installment_date",
            "foreclosed_after",
            "disposition_date",
        ],
        date_parser=dateparse,
    )
    return df


def load_acquisition_csv(lib, acquisition_path):
    """Loads acquisition data

    Returns
    -------
    Legate DataFrame
    """

    cols = [
        "loan_id",
        "orig_channel",
        "seller_name",
        "orig_interest_rate",
        "orig_upb",
        "orig_loan_term",
        "orig_date",
        "first_pay_date",
        "orig_ltv",
        "orig_cltv",
        "num_borrowers",
        "dti",
        "borrower_credit_score",
        "first_home_buyer",
        "loan_purpose",
        "property_type",
        "num_units",
        "occupancy_status",
        "property_state",
        "zip",
        "mortgage_insurance_percent",
        "product_type",
        "coborrow_credit_score",
        "mortgage_insurance_type",
        "relocation_mortgage_indicator",
        "quarter",
    ]

    dtypes = OrderedDict(
        [
            ("loan_id", "int64"),
            ("orig_channel", "category"),
            ("seller_name", "category"),
            ("orig_interest_rate", "float64"),
            ("orig_upb", "int64"),
            ("orig_loan_term", "int64"),
            ("orig_date", "date"),
            ("first_pay_date", "date"),
            ("orig_ltv", "float64"),
            ("orig_cltv", "float64"),
            ("num_borrowers", "float64"),
            ("dti", "float64"),
            ("borrower_credit_score", "float64"),
            ("first_home_buyer", "category"),
            ("loan_purpose", "category"),
            ("property_type", "category"),
            ("num_units", "int64"),
            ("occupancy_status", "category"),
            ("property_state", "category"),
            ("zip", "int64"),
            ("mortgage_insurance_percent", "float64"),
            ("product_type", "category"),
            ("coborrow_credit_score", "float64"),
            ("mortgage_insurance_type", "float64"),
            ("relocation_mortgage_indicator", "category"),
            ("quarter", "int32"),
        ]
    )

    def dateparse(x):
        return datetime.strptime(x, "%m/%Y")

    df = lib.read_csv(
        acquisition_path,
        names=cols,
        delimiter="|",
        dtype=dtypes,
        parse_dates=["orig_date", "first_pay_date"],
        date_parser=dateparse,
    )
    return df


def load_names(lib, col_names_filename):
    """Loads names used for renaming the banks

    Returns
    -------
    Legate DataFrame
    """

    cols = ["seller_name", "new"]

    dtypes = OrderedDict([("seller_name", "category"), ("new", "category")])

    return lib.read_csv(
        col_names_filename, names=cols, delimiter="|", dtype=dtypes, skiprows=1
    )


def create_ever_features(lib, df):
    everdf = df[["loan_id", "current_loan_delinquency_status"]]
    everdf = lib.group_and_apply(everdf, "loan_id", "max")
    del df
    everdf["ever_30"] = (
        everdf["current_loan_delinquency_status"] >= 1
    ).astype("int8")
    everdf["ever_90"] = (
        everdf["current_loan_delinquency_status"] >= 3
    ).astype("int8")
    everdf["ever_180"] = (
        everdf["current_loan_delinquency_status"] >= 6
    ).astype("int8")
    del everdf["current_loan_delinquency_status"]
    return everdf


def create_delinq_features(lib, df):
    delinq_df = df[
        [
            "loan_id",
            "monthly_reporting_period",
            "current_loan_delinquency_status",
        ]
    ]
    del df

    columns = ["loan_id", "monthly_reporting_period"]

    delinq_30 = lib.group_and_apply(
        delinq_df[columns][delinq_df.current_loan_delinquency_status >= 1],
        "loan_id",
        "min",
    )
    delinq_30["delinquency_30"] = delinq_30["monthly_reporting_period"]
    del delinq_30["monthly_reporting_period"]

    delinq_90 = lib.group_and_apply(
        delinq_df[columns][delinq_df.current_loan_delinquency_status >= 3],
        "loan_id",
        "min",
    )
    delinq_90["delinquency_90"] = delinq_90["monthly_reporting_period"]
    del delinq_90["monthly_reporting_period"]

    delinq_180 = lib.group_and_apply(
        delinq_df[columns][delinq_df.current_loan_delinquency_status >= 6],
        "loan_id",
        "min",
    )
    delinq_180["delinquency_180"] = delinq_180["monthly_reporting_period"]
    del delinq_180["monthly_reporting_period"]
    del delinq_df

    delinq_merge = delinq_30.merge(delinq_90, how="left", on=["loan_id"])
    delinq_merge["delinquency_90"] = delinq_merge["delinquency_90"].fillna(
        default_ts
    )

    delinq_merge = delinq_merge.merge(delinq_180, how="left", on=["loan_id"])
    delinq_merge["delinquency_180"] = delinq_merge["delinquency_180"].fillna(
        default_ts
    )
    del delinq_30
    del delinq_90
    del delinq_180

    return delinq_merge


def join_ever_delinq_features(everdf_tmp, delinq_merge):
    everdf = everdf_tmp.merge(delinq_merge, how="left", on=["loan_id"])
    del everdf_tmp
    del delinq_merge
    everdf["delinquency_30"] = everdf["delinquency_30"].fillna(default_ts)
    everdf["delinquency_90"] = everdf["delinquency_90"].fillna(default_ts)
    everdf["delinquency_180"] = everdf["delinquency_180"].fillna(default_ts)
    return everdf


def create_joined_df(df, everdf):
    test = df[
        [
            "loan_id",
            "monthly_reporting_period",
            "current_loan_delinquency_status",
            "current_actual_upb",
        ]
    ]
    del df
    test["timestamp"] = test["monthly_reporting_period"]
    del test["monthly_reporting_period"]
    test["timestamp_month"] = test["timestamp"].dt.month
    test["timestamp_year"] = test["timestamp"].dt.year
    test["delinquency_12"] = test["current_loan_delinquency_status"]
    del test["current_loan_delinquency_status"]
    test["upb_12"] = test["current_actual_upb"]
    del test["current_actual_upb"]
    test["upb_12"] = test["upb_12"].fillna(999999999)
    test["delinquency_12"] = test["delinquency_12"].fillna(-1)

    joined_df = test.merge(everdf, how="left", on=["loan_id"])
    del everdf
    del test

    joined_df["ever_30"] = joined_df["ever_30"].fillna(-1)
    joined_df["ever_90"] = joined_df["ever_90"].fillna(-1)
    joined_df["ever_180"] = joined_df["ever_180"].fillna(-1)
    joined_df["delinquency_30"] = joined_df["delinquency_30"].fillna(-1)
    joined_df["delinquency_90"] = joined_df["delinquency_90"].fillna(-1)
    joined_df["delinquency_180"] = joined_df["delinquency_180"].fillna(-1)

    joined_df["timestamp_year"] = joined_df["timestamp_year"].astype("int32")
    joined_df["timestamp_month"] = joined_df["timestamp_month"].astype("int32")

    return joined_df


def create_12_mon_features(lib, joined_df):
    testdfs = []
    n_months = 12
    for y in range(1, n_months + 1):
        tmpdf = joined_df[
            [
                "loan_id",
                "timestamp_year",
                "timestamp_month",
                "delinquency_12",
                "upb_12",
            ]
        ]
        tmpdf["josh_months"] = (
            tmpdf["timestamp_year"] * 12 + tmpdf["timestamp_month"]
        )
        tmpdf["josh_mody_n"] = (
            (tmpdf["josh_months"].astype("float64") - 24000 - y) / 12
        ).astype("int64")
        tmpdf = lib.group_and_apply(
            tmpdf,
            ["loan_id", "josh_mody_n"],
            "agg",
            {"delinquency_12": "max", "upb_12": "min"},
        )
        tmpdf["delinquency_12"] = (tmpdf["delinquency_12"] > 3).astype("int32")
        tmpdf["delinquency_12"] += (tmpdf["upb_12"] == 0).astype("int32")
        tmpdf["timestamp_year"] = (
            ((tmpdf["josh_mody_n"] * n_months) + 24000 + (y - 1)) / 12
        ).astype("int16")
        tmpdf["timestamp_month"] = np.int8(y)
        del tmpdf["josh_mody_n"]
        testdfs.append(tmpdf)
        del tmpdf
    del joined_df
    return lib.concat(testdfs)


def combine_joined_12_mon(joined_df, testdf):
    del joined_df["delinquency_12"]
    del joined_df["upb_12"]
    joined_df["timestamp_year"] = joined_df["timestamp_year"].astype("int16")
    joined_df["timestamp_month"] = joined_df["timestamp_month"].astype("int8")
    result = joined_df.merge(
        testdf, how="left", on=["loan_id", "timestamp_year", "timestamp_month"]
    )
    return result


def final_performance_delinquency(merged, joined_df, use_null_workaround):
    if use_null_workaround:
        merged = null_workaround(merged)
        joined_df = null_workaround(joined_df)
    merged["timestamp_month"] = merged["monthly_reporting_period"].dt.month
    merged["timestamp_month"] = merged["timestamp_month"].astype("int8")
    merged["timestamp_year"] = merged["monthly_reporting_period"].dt.year
    merged["timestamp_year"] = merged["timestamp_year"].astype("int16")
    merged = merged.merge(
        joined_df,
        how="left",
        on=["loan_id", "timestamp_year", "timestamp_month"],
    )
    del merged["timestamp_year"]
    del merged["timestamp_month"]
    return merged


def join_perf_acq_dfs(perf, acq, use_null_workaround):
    if use_null_workaround:
        perf = null_workaround(perf)
        acq = null_workaround(acq)
    return perf.merge(acq, how="left", on="loan_id")


def run_workflow(
    lib,
    num_procs,
    col_names_filename,
    acq_filenames,
    perf_filenames,
    use_null_workaround,
    shuffle_first,
):
    print("Input files:", flush=True)
    for filename in acq_filenames:
        print("  - " + filename, flush=True)
    for filename in perf_filenames:
        print("  - " + filename, flush=True)
    print(flush=True)

    lib.stopwatch()

    names = lib.persist(load_names(lib, col_names_filename))
    acq_df = lib.persist(load_acquisition_csv(lib, acq_filenames))
    perf_df_tmp = lib.persist(load_performance_csv(lib, perf_filenames))

    lib.wait_for(names, acq_df, perf_df_tmp)
    time_csv = lib.stopwatch()
    print("CSV parsing: %s s" % time_csv, flush=True)

    acq_df = acq_df.merge(names, how="left", on="seller_name")
    del acq_df["seller_name"]
    acq_df["seller_name"] = acq_df["new"]
    del acq_df["new"]
    del names

    df = perf_df_tmp
    if shuffle_first:
        df = df._shuffle("loan_id")

    everdf = create_ever_features(lib, df)

    delinq_merge = create_delinq_features(lib, df)
    everdf = join_ever_delinq_features(everdf, delinq_merge)
    del delinq_merge

    joined_df = create_joined_df(df, everdf)
    testdf = create_12_mon_features(lib, joined_df)
    joined_df = combine_joined_12_mon(joined_df, testdf)
    del testdf

    perf_df = final_performance_delinquency(df, joined_df, use_null_workaround)
    del (df, joined_df)

    final_df = lib.persist(
        join_perf_acq_dfs(perf_df, acq_df, use_null_workaround)
    )
    del perf_df
    del acq_df

    lib.wait_for(final_df)
    time_processing = lib.stopwatch()
    print("Processing: %s s" % time_processing, flush=True)

    print("Elapsed time: %s s" % (time_csv + time_processing), flush=True)

    return final_df


def test(
    num_procs=1,
    num_runs=1,
    input_path="benchmarks/mortgage/data",
    use_null_workaround=False,
    shuffle_first=False,
    package="legate",
):
    if not os.path.exists(input_path):
        raise Exception(
            f"'{input_path}' does not exist! Please pass a valid path "
            "to the mortgage dataset."
        )

    class LibraryBase:
        def __init__(self):
            self._timestamp = None

        def stopwatch(self):
            prev_ts = self._timestamp
            self._timestamp = time.time()
            return 0 if prev_ts is None else self._timestamp - prev_ts

    if package == "pandas":
        print("Using pandas", flush=True)

        class Library(LibraryBase):
            def group_and_apply(self, df, cols, func, *args, **kwargs):
                grouped = df.groupby(cols, as_index=False, sort=False)
                return getattr(grouped, func)(*args, **kwargs)

            def read_csv(self, files, **kwargs):
                if type(files) == str:
                    files = [files]
                if "dtype" in kwargs:
                    kwargs["dtype"] = OrderedDict(
                        [
                            (col, ("str" if dtype == "date" else dtype))
                            for (col, dtype) in kwargs["dtype"].items()
                        ]
                    )
                return self.concat([pd.read_csv(f, **kwargs) for f in files])

            def concat(self, dfs, **kwargs):
                kwargs["ignore_index"] = True
                return pd.concat(dfs, **kwargs)

            def persist(self, df):
                return df

            def wait_for(self, *args):
                pass

    elif package == "cudf":
        import cudf

        print("Using cudf", flush=True)

        class Library(LibraryBase):
            def group_and_apply(self, df, cols, func, *args, **kwargs):
                grouped = df.groupby(cols, as_index=False, sort=False)
                return getattr(grouped, func)(*args, **kwargs)

            def read_csv(self, files, **kwargs):
                if type(files) == str:
                    files = [files]
                if "dtype" in kwargs:
                    kwargs["dtype"] = OrderedDict(
                        [
                            (col, ("str" if dtype == "category" else dtype))
                            for (col, dtype) in kwargs["dtype"].items()
                        ]
                    )
                return self.concat([cudf.read_csv(f, **kwargs) for f in files])

            def concat(self, dfs, **kwargs):
                kwargs["ignore_index"] = True
                return cudf.concat(dfs, **kwargs)

            def persist(self, df):
                return df

            def wait_for(self, *args):
                pass

    elif package == "legate":
        from legate.core import get_legion_context, get_legion_runtime, legion

        import legate.pandas

        print("Using legate.pandas", flush=True)

        class Library(LibraryBase):
            def group_and_apply(self, df, cols, func, *args, **kwargs):
                grouped = df.groupby(cols, as_index=False, sort=False)
                return getattr(grouped, func)(*args, **kwargs)

            def read_csv(self, files, **kwargs):
                return legate.pandas.read_csv(files, **kwargs)

            def concat(self, dfs, **kwargs):
                return legate.pandas.concat(dfs, **kwargs)

            def persist(self, df):
                return df

            def wait_for(self, *args):
                runtime = get_legion_runtime()
                context = get_legion_context()
                future = legion.legion_runtime_issue_execution_fence(
                    runtime, context
                )
                legion.legion_future_get_void_result(future)

    elif package == "dask":
        import dask.dataframe
        from dask.distributed import Client, LocalCluster, wait

        if "SCHEDULER_FILE" in os.environ:
            print("Using dask.dataframe with preexisting cluster", flush=True)
            Client(scheduler_file=os.environ["SCHEDULER_FILE"])
        else:
            print("Using dask.dataframe with LocalCluster", flush=True)
            Client(LocalCluster())

        class Library(LibraryBase):
            def group_and_apply(self, df, cols, func, *args, **kwargs):
                grouped = df.groupby(cols, sort=False)
                return getattr(grouped, func)(*args, **kwargs).reset_index()

            def read_csv(self, files, **kwargs):
                if "dtype" in kwargs:
                    kwargs["dtype"] = OrderedDict(
                        [
                            (col, ("str" if dtype == "date" else dtype))
                            for (col, dtype) in kwargs["dtype"].items()
                        ]
                    )
                kwargs["blocksize"] = None
                return dask.dataframe.read_csv(files, **kwargs)

            def concat(self, dfs, **kwargs):
                return dask.dataframe.concat(dfs, **kwargs)

            def persist(self, df):
                return df.persist()

            def wait_for(self, *args):
                wait([*args])

    elif package == "dask_cudf":
        import dask_cudf
        from dask.distributed import Client, wait

        if "SCHEDULER_FILE" in os.environ:
            print("Using dask_cudf with preexisting cluster", flush=True)
            Client(scheduler_file=os.environ["SCHEDULER_FILE"])
        else:
            print("Using dask_cudf with LocalCUDACluster", flush=True)
            from dask_cuda import LocalCUDACluster

            Client(
                LocalCUDACluster(
                    rmm_pool_size="10G",
                )
            )

        class Library(LibraryBase):
            def group_and_apply(self, df, cols, func, *args, **kwargs):
                grouped = df.groupby(cols, sort=False)
                return getattr(grouped, func)(*args, **kwargs).reset_index()

            def read_csv(self, files, **kwargs):
                if "dtype" in kwargs:
                    kwargs["dtype"] = OrderedDict(
                        [
                            (col, ("str" if dtype == "category" else dtype))
                            for (col, dtype) in kwargs["dtype"].items()
                        ]
                    )
                kwargs["chunksize"] = None
                return dask_cudf.read_csv(files, **kwargs)

            def concat(self, dfs, **kwargs):
                return dask_cudf.concat(dfs, **kwargs)

            def persist(self, df):
                return df.persist()

            def wait_for(self, *args):
                wait([*args])

    else:
        raise ValueError("Unsupported library %s" % package)
    lib = Library()

    col_names_filename = os.path.join(input_path, "names.csv")
    perf_filenames = (
        glob.glob(os.path.join(input_path, "perf", "Performance_20??Q?.txt"))
        + glob.glob(
            os.path.join(input_path, "perf", "Performance_20??Q?.txt_0")
        )
        + glob.glob(
            os.path.join(input_path, "perf", "Performance_20??Q?.txt_0_0")
        )
    )

    perf_filenames = sorted(perf_filenames, key=lambda f: os.stat(f).st_size)
    acq_filenames = [
        f.replace("Performance", "Acquisition")
        .replace("perf", "acq")
        .replace("_0", "")
        for f in perf_filenames
    ]

    num_procs = min(
        max(num_procs, 1), min(len(acq_filenames), len(perf_filenames))
    )

    for run_id in range(num_runs):
        print(f"[Run {run_id}]")
        result = run_workflow(
            lib,
            num_procs,
            col_names_filename,
            acq_filenames[:num_procs],
            perf_filenames[:num_procs],
            use_null_workaround,
            shuffle_first,
        )
        del result


def driver():
    parser = argparse.ArgumentParser(description="Mortgage data benchmark")

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
        "--input_path",
        dest="input_path",
        type=str,
        default="benchmarks/mortgage/data",
        help="Path to the input data",
    )

    parser.add_argument(
        "--null_workaround",
        dest="use_null_workaround",
        action="store_true",
        required=False,
        default=False,
        help="Replace all nulls with -1",
    )

    parser.add_argument(
        "--shuffle_first",
        dest="shuffle_first",
        action="store_true",
        required=False,
        default=False,
        help="Hash partition the performance table by loan ids first",
    )

    parser.add_argument(
        "--package",
        dest="package",
        type=str,
        choices=["pandas", "cudf", "legate", "dask", "dask_cudf"],
        default="legate",
        help="Dataframe package to use",
    )

    args = parser.parse_args()

    test(**vars(args))


if __name__ == "__main__":
    driver()

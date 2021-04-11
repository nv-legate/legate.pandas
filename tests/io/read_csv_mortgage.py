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
from collections import OrderedDict
from datetime import datetime

import pandas as pd

from legate import pandas as lp
from tests.utils import equals

col_names_path = os.path.join(
    os.path.dirname(__file__), "files", "mortgage", "names.csv"
)
acq_data_path = os.path.join(
    os.path.dirname(__file__), "files", "mortgage", "acq.csv"
)
perf_data_path = os.path.join(
    os.path.dirname(__file__), "files", "mortgage", "perf.csv"
)


def load_names():
    """Loads names used for renaming the banks

    Returns
    -------
    Legate DataFrame
    """

    cols = ["seller_name", "new"]

    dtypes = OrderedDict([("seller_name", "category"), ("new", "category")])

    out_pd = pd.read_csv(
        col_names_path,
        names=cols,
        delimiter="|",
        dtype=dtypes,
        skiprows=1,
    )

    out_lp = lp.read_csv(
        col_names_path,
        names=cols,
        delimiter="|",
        dtype=dtypes,
        skiprows=1,
    )

    assert equals(out_lp, out_pd)


def load_acquisition_csv():
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
            ("orig_date", "str"),
            ("first_pay_date", "str"),
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

    out_pd = pd.read_csv(
        acq_data_path,
        names=cols,
        delimiter="|",
        index_col=False,
        dtype=dtypes,
        parse_dates=["orig_date", "first_pay_date"],
        date_parser=dateparse,
    )

    out_lp = lp.read_csv(
        acq_data_path,
        names=cols,
        delimiter="|",
        index_col=False,
        dtype=dtypes,
        parse_dates=["orig_date", "first_pay_date"],
    )

    assert equals(out_lp, out_pd)


def load_performance_csv():
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
            ("monthly_reporting_period", "str"),
            ("servicer", "category"),
            ("interest_rate", "float64"),
            ("current_actual_upb", "float64"),
            ("loan_age", "float64"),
            ("remaining_months_to_legal_maturity", "float64"),
            ("adj_remaining_months_to_maturity", "float64"),
            ("maturity_date", "str"),
            ("msa", "float64"),
            ("current_loan_delinquency_status", "int32"),
            ("mod_flag", "category"),
            ("zero_balance_code", "category"),
            ("zero_balance_effective_date", "str"),
            ("last_paid_installment_date", "str"),
            ("foreclosed_after", "str"),
            ("disposition_date", "str"),
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

    out_pd = pd.read_csv(
        perf_data_path,
        names=cols,
        delimiter="|",
        index_col=False,
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

    out_lp = lp.read_csv(
        perf_data_path,
        names=cols,
        delimiter="|",
        index_col=False,
        dtype=dtypes,
        parse_dates=[
            "monthly_reporting_period",
            "maturity_date",
            "zero_balance_effective_date",
            "last_paid_installment_date",
            "foreclosed_after",
            "disposition_date",
        ],
    )

    assert equals(out_lp, out_pd)


names = load_names()
acq_df = load_acquisition_csv()
perf_df_tmp = load_performance_csv()

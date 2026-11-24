import re

import duckdb
import pandas as pd
import pytest

from credit_risk.config import get_paths
from credit_risk.db import read_sql, run_sql_file
from credit_risk.features import FEATURE_QUERIES

ALLOWED_SOURCES = {"application_clean", "installments_valid", "bureau_valid", "bureau_balance_valid",
                   "previous_application_valid", "pos_cash_valid", "credit_card_valid"}
RAW_BEHAVIOR_TABLES = {"installments_payments", "bureau", "bureau_balance", "previous_application", "pos_cash_balance", "credit_card_balance"}
BEHAVIOR_GROUPS = ("installments", "bureau", "bureau_balance", "previous_application", "pos_cash", "credit_card")
PREVIOUS_DATES = ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION"]
VIEW_SOURCES = {
    "installments_valid": "installments_payments",
    "bureau_valid": "bureau",
    "bureau_balance_valid": "bureau_balance",
    "previous_application_valid": "previous_application",
    "pos_cash_valid": "pos_cash_balance",
    "credit_card_valid": "credit_card_balance",
}


def referenced_relations(sql: str) -> set[str]:
    ctes = set(re.findall(r"(\w+)\s+AS\s*\(", sql, flags=re.IGNORECASE))
    sources = set(re.findall(r"\b(?:FROM|JOIN)\s+(\w+)", sql, flags=re.IGNORECASE))
    return sources - ctes


@pytest.mark.parametrize("query_file", FEATURE_QUERIES.values())
def test_feature_queries_read_only_filtered_views(query_file):
    relations = referenced_relations(read_sql(query_file))
    assert relations <= ALLOWED_SOURCES, f"{query_file} reads {relations - ALLOWED_SOURCES}"
    assert not relations & RAW_BEHAVIOR_TABLES


def test_valid_views_filter_every_day_offset():
    sql = (get_paths().sql / "02_valid_views.sql").read_text()
    for condition in ("DAYS_INSTALMENT <= 0", "DAYS_ENTRY_PAYMENT, 0) <= 0", "DAYS_CREDIT <= 0",
                      "DAYS_ENDDATE_FACT, 0) <= 0", "DAYS_CREDIT_UPDATE, 0) <= 0", "MONTHS_BALANCE <= 0", "DAYS_DECISION <= 0",
                      *[f"CASE WHEN {column} <= 0" for column in PREVIOUS_DATES]):
        assert condition in sql


def synthetic_tables() -> dict[str, pd.DataFrame]:
    installments = pd.DataFrame({
        "SK_ID_PREV": [1, 1, 1, 2, 2, 3],
        "SK_ID_CURR": [10, 10, 10, 10, 10, 20],
        "NUM_INSTALMENT_VERSION": [1, 1, 1, 1, 1, 1],
        "NUM_INSTALMENT_NUMBER": [1, 2, 3, 1, 2, 1],
        "DAYS_INSTALMENT": [-400.0, -100.0, 30.0, -50.0, -20.0, -10.0],
        "DAYS_ENTRY_PAYMENT": [-390.0, -60.0, 45.0, -50.0, 5.0, -12.0],
        "AMT_INSTALMENT": [100.0, 100.0, 100.0, 50.0, 50.0, 80.0],
        "AMT_PAYMENT": [100.0, 90.0, 5.0, 50.0, 50.0, 80.0],
    })
    bureau = pd.DataFrame({
        "SK_ID_CURR": [10, 10, 10, 20],
        "SK_ID_BUREAU": [100, 101, 102, 200],
        "CREDIT_ACTIVE": ["Active", "Closed", "Active", "Active"],
        "CREDIT_TYPE": ["Consumer credit", "Car loan", "Mortgage", "Consumer credit"],
        "DAYS_CREDIT": [-300.0, -900.0, 15.0, -40.0],
        "DAYS_ENDDATE_FACT": [None, -100.0, None, 3.0],
        "DAYS_CREDIT_UPDATE": [-5.0, -90.0, 2.0, -1.0],
        "CREDIT_DAY_OVERDUE": [0, 0, 60, 10],
        "AMT_CREDIT_MAX_OVERDUE": [0.0, 10.0, 999.0, 5.0],
        "AMT_CREDIT_SUM_OVERDUE": [0.0, 0.0, 500.0, 1.0],
        "AMT_CREDIT_SUM": [1000.0, 500.0, 9000.0, 300.0],
        "AMT_CREDIT_SUM_DEBT": [400.0, 0.0, 9000.0, 100.0],
        "AMT_CREDIT_SUM_LIMIT": [0.0, 0.0, 0.0, 0.0],
        "CNT_CREDIT_PROLONG": [0, 1, 3, 0],
    })
    bureau_balance = pd.DataFrame({
        "SK_ID_BUREAU": [100, 100, 100, 100, 101, 101, 102, 200],
        "MONTHS_BALANCE": [-3, -2, -1, 1, -20, -19, -1, -1],
        "STATUS": ["0", "1", "2", "5", "C", "X", "5", "0"],
    })
    previous = pd.DataFrame({
        "SK_ID_PREV": [1, 2, 3, 4],
        "SK_ID_CURR": [10, 10, 10, 20],
        "NAME_CONTRACT_TYPE": ["Cash loans", "Consumer loans", "Revolving loans", "Cash loans"],
        "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved", "Approved"],
        "NAME_YIELD_GROUP": ["high", "low_normal", "middle", "high"],
        "NAME_CLIENT_TYPE": ["Repeater", "New", "Repeater", "New"],
        "AMT_APPLICATION": [1000.0, 500.0, 0.0, 200.0],
        "AMT_CREDIT": [1100.0, 0.0, 300.0, 200.0],
        "AMT_ANNUITY": [100.0, None, 30.0, 20.0],
        "RATE_DOWN_PAYMENT": [0.1, None, 0.0, 0.2],
        "CNT_PAYMENT": [12.0, None, 12.0, 10.0],
        "DAYS_DECISION": [-500.0, -100.0, -30.0, 5.0],
        "DAYS_FIRST_DRAWING": [365243.0, None, -20.0, 10.0],
        "DAYS_FIRST_DUE": [-470.0, None, -5.0, 30.0],
        "DAYS_LAST_DUE_1ST_VERSION": [-140.0, None, 300.0, 300.0],
        "DAYS_LAST_DUE": [-150.0, None, 365243.0, 365243.0],
        "DAYS_TERMINATION": [-140.0, None, 330.0, 365243.0],
    })
    pos_cash = pd.DataFrame({
        "SK_ID_PREV": [1, 1, 1, 5],
        "SK_ID_CURR": [10, 10, 10, 20],
        "MONTHS_BALANCE": [-3, -2, 2, -1],
        "CNT_INSTALMENT": [12.0, 12.0, 12.0, 6.0],
        "CNT_INSTALMENT_FUTURE": [5.0, 4.0, 0.0, 3.0],
        "NAME_CONTRACT_STATUS": ["Active", "Active", "Completed", "Active"],
        "SK_DPD": [0, 7, 90, 0],
        "SK_DPD_DEF": [0, 0, 90, 0],
    })
    credit_card = pd.DataFrame({
        "SK_ID_PREV": [7, 7, 7, 8],
        "SK_ID_CURR": [10, 10, 10, 20],
        "MONTHS_BALANCE": [-2, -1, 1, -1],
        "AMT_BALANCE": [500.0, 900.0, 5000.0, 0.0],
        "AMT_CREDIT_LIMIT_ACTUAL": [1000.0, 1000.0, 1000.0, 0.0],
        "AMT_DRAWINGS_ATM_CURRENT": [100.0, 0.0, 900.0, None],
        "CNT_DRAWINGS_CURRENT": [1, 0, 9, 0],
        "AMT_PAYMENT_TOTAL_CURRENT": [50.0, 10.0, 0.0, 0.0],
        "AMT_INST_MIN_REGULARITY": [40.0, 45.0, 250.0, 0.0],
        "SK_DPD": [0, 0, 60, 0],
        "SK_DPD_DEF": [0, 0, 60, 0],
    })
    return {"installments_payments": installments, "bureau": bureau, "bureau_balance": bureau_balance,
            "previous_application": previous, "pos_cash_balance": pos_cash, "credit_card_balance": credit_card}


def compute_features(tables: dict[str, pd.DataFrame], through_filter: bool) -> dict[str, pd.DataFrame]:
    con = duckdb.connect()
    for name, frame in tables.items():
        con.register(f"{name}_frame", frame)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_frame")
    if through_filter:
        run_sql_file(con, "02_valid_views.sql")
    else:
        for view, source in VIEW_SOURCES.items():
            con.execute(f"CREATE VIEW {view} AS SELECT * FROM {source}")
    results = {}
    for group in BEHAVIOR_GROUPS:
        results[group] = con.execute(read_sql(FEATURE_QUERIES[group]) + " ORDER BY SK_ID_CURR").df()
    return results


def remove_future_rows(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    installments = tables["installments_payments"]
    installments = installments[(installments.DAYS_INSTALMENT <= 0) & (installments.DAYS_ENTRY_PAYMENT.fillna(0) <= 0)]
    bureau = tables["bureau"]
    bureau = bureau[(bureau.DAYS_CREDIT <= 0) & (bureau.DAYS_ENDDATE_FACT.fillna(0) <= 0) & (bureau.DAYS_CREDIT_UPDATE.fillna(0) <= 0)]
    balance = tables["bureau_balance"]
    balance = balance[(balance.MONTHS_BALANCE <= 0) & balance.SK_ID_BUREAU.isin(bureau.SK_ID_BUREAU)]
    previous = tables["previous_application"]
    previous = previous[previous.DAYS_DECISION <= 0].copy()
    for column in PREVIOUS_DATES:
        previous[column] = previous[column].where(previous[column] <= 0)
    pos_cash = tables["pos_cash_balance"]
    credit_card = tables["credit_card_balance"]
    return {"installments_payments": installments, "bureau": bureau, "bureau_balance": balance, "previous_application": previous,
            "pos_cash_balance": pos_cash[pos_cash.MONTHS_BALANCE <= 0], "credit_card_balance": credit_card[credit_card.MONTHS_BALANCE <= 0]}


def test_future_rows_never_reach_features():
    tables = synthetic_tables()
    filtered = compute_features(tables, through_filter=True)
    expected = compute_features(remove_future_rows(tables), through_filter=False)
    for group in filtered:
        pd.testing.assert_frame_equal(filtered[group], expected[group], check_dtype=False)


def test_future_rows_would_change_features_without_filter():
    tables = synthetic_tables()
    filtered = compute_features(tables, through_filter=True)
    unfiltered = compute_features(tables, through_filter=False)
    for group in BEHAVIOR_GROUPS:
        assert not filtered[group].equals(unfiltered[group]), group


def test_installment_dpd_and_windows():
    features = compute_features(synthetic_tables(), through_filter=True)["installments"].set_index("SK_ID_CURR")
    applicant = features.loc[10]
    assert applicant["INST_COUNT_ALL"] == 3
    assert applicant["INST_COUNT_3M"] == 1
    assert applicant["INST_MAX_DPD_ALL"] == 40
    assert applicant["INST_LATE_COUNT_ALL"] == 2
    assert applicant["INST_DPD30_COUNT_ALL"] == 1
    assert applicant["INST_TOTAL_UNDERPAYMENT_ALL"] == pytest.approx(10.0)
    assert applicant["INST_DAYS_SINCE_LAST_LATE"] == 100


def test_previous_application_and_balance_features():
    features = compute_features(synthetic_tables(), through_filter=True)
    previous = features["previous_application"].set_index("SK_ID_CURR")
    assert list(previous.index) == [10]
    assert previous.loc[10, "PREV_REFUSED_COUNT"] == 1
    assert previous.loc[10, "PREV_OPEN_LOAN_COUNT"] == 1
    assert previous.loc[10, "PREV_DAYS_SINCE_LAST_DECISION"] == 30
    pos_cash = features["pos_cash"].set_index("SK_ID_CURR")
    assert pos_cash.loc[10, "POS_MAX_DPD"] == 7
    assert pos_cash.loc[10, "POS_REMAINING_INSTALMENTS"] == 4
    credit_card = features["credit_card"].set_index("SK_ID_CURR")
    assert credit_card.loc[10, "CC_MAX_UTILIZATION"] == pytest.approx(0.9)
    assert credit_card.loc[10, "CC_BELOW_MINIMUM_SHARE"] == pytest.approx(0.5)
    assert pd.isna(credit_card.loc[20, "CC_MEAN_UTILIZATION"])

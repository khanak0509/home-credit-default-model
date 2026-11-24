from typing import Any

import duckdb

from credit_risk.artifacts import markdown_table, write_json
from credit_risk.config import get_paths, load_config
from credit_risk.db import connect
from credit_risk.features import leakage_filter_counts, prepare_views

KEY_COLUMNS = {
    "application_train": ["SK_ID_CURR"],
    "bureau": ["SK_ID_BUREAU"],
    "bureau_balance": ["SK_ID_BUREAU", "MONTHS_BALANCE"],
    "installments_payments": ["SK_ID_PREV", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER", "DAYS_ENTRY_PAYMENT", "AMT_PAYMENT"],
    "previous_application": ["SK_ID_PREV"],
    "pos_cash_balance": ["SK_ID_PREV", "MONTHS_BALANCE"],
    "credit_card_balance": ["SK_ID_PREV", "MONTHS_BALANCE"],
}

ID_COLUMNS = {"SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "TARGET"}
NUMERIC_TYPES = ("BIGINT", "INTEGER", "DOUBLE", "FLOAT", "DECIMAL", "SMALLINT", "TINYINT", "HUGEINT")


def table_columns(con: duckdb.DuckDBPyConnection, table: str) -> list[tuple[str, str]]:
    return [(row[0], row[1]) for row in con.execute(f"DESCRIBE {table}").fetchall()]


def is_numeric(dtype: str) -> bool:
    return dtype.startswith(NUMERIC_TYPES)


def column_profile(con: duckdb.DuckDBPyConnection, table: str, column: str, dtype: str, rows: int) -> dict[str, Any]:
    missing, distinct = con.execute(
        f'SELECT count(*) FILTER (WHERE "{column}" IS NULL), approx_count_distinct("{column}") FROM {table}'
    ).fetchone()
    top_count = con.execute(
        f'SELECT max(n) FROM (SELECT count(*) AS n FROM {table} GROUP BY "{column}")'
    ).fetchone()[0]
    return {
        "dtype": dtype,
        "missing": missing,
        "missing_rate": missing / rows,
        "distinct": distinct,
        "top_value_share": top_count / rows,
    }


def iqr_profile(con: duckdb.DuckDBPyConnection, table: str, column: str, multiplier: float) -> dict[str, Any] | None:
    q1, q3, low, high = con.execute(
        f'SELECT quantile_cont("{column}", 0.25), quantile_cont("{column}", 0.75), min("{column}"), max("{column}") FROM {table}'
    ).fetchone()
    if q1 is None:
        return None
    spread = q3 - q1
    lower, upper = q1 - multiplier * spread, q3 + multiplier * spread
    outliers = con.execute(
        f'SELECT count(*) FROM {table} WHERE "{column}" < {lower} OR "{column}" > {upper}'
    ).fetchone()[0]
    return {"q1": q1, "q3": q3, "iqr": spread, "lower_fence": lower, "upper_fence": upper, "min": low, "max": high, "outliers": outliers}


def duplicate_keys(con: duckdb.DuckDBPyConnection, table: str) -> dict[str, Any]:
    keys = ", ".join(KEY_COLUMNS[table])
    duplicated = con.execute(
        f"SELECT coalesce(sum(n - 1), 0) FROM (SELECT count(*) AS n FROM {table} GROUP BY {keys} HAVING count(*) > 1)"
    ).fetchone()[0]
    return {"key": KEY_COLUMNS[table], "duplicate_rows": int(duplicated)}


def profile_table(con: duckdb.DuckDBPyConnection, table: str, config: dict[str, Any]) -> dict[str, Any]:
    quality = config["quality"]
    rows = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    columns, outliers = {}, {}
    for column, dtype in table_columns(con, table):
        profile = column_profile(con, table, column, dtype, rows)
        profile["constant"] = profile["distinct"] <= 1
        profile["near_constant"] = not profile["constant"] and profile["top_value_share"] >= quality["near_constant_share"]
        columns[column] = profile
        if is_numeric(dtype) and column not in ID_COLUMNS and profile["distinct"] > 2:
            summary = iqr_profile(con, table, column, quality["iqr_multiplier"])
            if summary is not None:
                summary["outlier_rate"] = summary["outliers"] / (rows - profile["missing"])
                outliers[column] = summary
    return {"rows": rows, "columns": columns, "outliers": outliers, "duplicates": duplicate_keys(con, table)}


def scalar(con: duckdb.DuckDBPyConnection, query: str) -> Any:
    return con.execute(query).fetchone()[0]


def known_anomalies(con: duckdb.DuckDBPyConnection, config: dict[str, Any]) -> list[dict[str, Any]]:
    sentinel = config["quality"]["days_employed_sentinel"]
    unknown_gender = config["quality"]["gender_unknown_value"]
    sentinel_rate, other_rate = con.execute(
        f"SELECT avg(TARGET) FILTER (WHERE DAYS_EMPLOYED = {sentinel}), avg(TARGET) FILTER (WHERE DAYS_EMPLOYED <> {sentinel}) FROM application_train"
    ).fetchone()
    checks = [
        ("application_train", "DAYS_EMPLOYED", f"sentinel value {sentinel} (about 1000 years of employment)",
         f"SELECT count(*) FROM application_train WHERE DAYS_EMPLOYED = {sentinel}", "converted to null, indicator DAYS_EMPLOYED_ANOMALY added"),
        ("application_train", "CODE_GENDER", f"placeholder value {unknown_gender}",
         f"SELECT count(*) FROM application_train WHERE CODE_GENDER = '{unknown_gender}'", "converted to null; column excluded from model features, kept for fairness audit"),
        ("application_train", "ORGANIZATION_TYPE", "placeholder value XNA",
         "SELECT count(*) FROM application_train WHERE ORGANIZATION_TYPE = 'XNA'", "kept as its own category; overlaps with the DAYS_EMPLOYED sentinel"),
        ("application_train", "NAME_FAMILY_STATUS", "placeholder value Unknown",
         "SELECT count(*) FROM application_train WHERE NAME_FAMILY_STATUS = 'Unknown'", "kept as its own category"),
        ("application_train", "AMT_INCOME_TOTAL", "income above 10 million",
         "SELECT count(*) FROM application_train WHERE AMT_INCOME_TOTAL > 1e7", "kept; tree models and WoE binning are insensitive to the scale"),
        ("bureau", "DAYS_CREDIT_ENDDATE", "below -40000 days (more than 100 years ago)",
         "SELECT count(*) FROM bureau WHERE DAYS_CREDIT_ENDDATE < -40000", "column not used in features"),
        ("bureau", "DAYS_ENDDATE_FACT", "below -40000 days",
         "SELECT count(*) FROM bureau WHERE DAYS_ENDDATE_FACT < -40000", "kept; column only used in the leakage filter"),
        ("bureau", "AMT_CREDIT_SUM_DEBT", "negative debt",
         "SELECT count(*) FROM bureau WHERE AMT_CREDIT_SUM_DEBT < 0", "kept; enters debt to credit ratio as reported"),
        ("bureau", "SK_ID_CURR", "applicant not in application_train (test set applicants)",
         "SELECT count(*) FROM bureau ANTI JOIN application_train USING (SK_ID_CURR)", "dropped implicitly by the left join onto application_train"),
        ("bureau_balance", "SK_ID_BUREAU", "no matching bureau record",
         "SELECT count(*) FROM bureau_balance ANTI JOIN bureau USING (SK_ID_BUREAU)", "dropped by the join to bureau"),
        ("installments_payments", "DAYS_ENTRY_PAYMENT", "missing payment date",
         "SELECT count(*) FROM installments_payments WHERE DAYS_ENTRY_PAYMENT IS NULL", "DPD left null for these installments; counted in INST_UNPAID_COUNT"),
        ("installments_payments", "AMT_INSTALMENT", "zero installment amount",
         "SELECT count(*) FROM installments_payments WHERE AMT_INSTALMENT = 0", "payment ratio left null (division guarded)"),
        ("previous_application", "DAYS_FIRST_DUE", f"sentinel value {sentinel} (event never happened)",
         f"SELECT count(*) FROM previous_application WHERE DAYS_FIRST_DUE = {sentinel}", "set to null in previous_application_valid"),
        ("previous_application", "AMT_APPLICATION", "zero amount requested",
         "SELECT count(*) FROM previous_application WHERE AMT_APPLICATION = 0", "credit to application ratio left null (division guarded)"),
        ("previous_application", "NAME_CONTRACT_STATUS", "Unused offer status",
         "SELECT count(*) FROM previous_application WHERE NAME_CONTRACT_STATUS = 'Unused offer'", "counted in totals, not in approved, refused or canceled"),
        ("pos_cash_balance", "CNT_INSTALMENT_FUTURE", "missing remaining installment count",
         "SELECT count(*) FROM pos_cash_balance WHERE CNT_INSTALMENT_FUTURE IS NULL", "ignored by the aggregates"),
        ("credit_card_balance", "AMT_DRAWINGS_ATM_CURRENT", "negative drawing amount",
         "SELECT count(*) FROM credit_card_balance WHERE AMT_DRAWINGS_ATM_CURRENT < 0", "kept as reported"),
        ("credit_card_balance", "AMT_CREDIT_LIMIT_ACTUAL", "zero credit limit",
         "SELECT count(*) FROM credit_card_balance WHERE AMT_CREDIT_LIMIT_ACTUAL = 0", "utilization left null (division guarded)"),
    ]
    anomalies = []
    for table, column, description, query, handling in checks:
        anomalies.append({"table": table, "column": column, "description": description, "rows": scalar(con, query), "handling": handling})
    anomalies[0]["default_rate_flagged"] = sentinel_rate
    anomalies[0]["default_rate_other"] = other_rate
    return anomalies


def transformations(con: duckdb.DuckDBPyConnection, config: dict[str, Any]) -> list[dict[str, Any]]:
    sentinel = config["quality"]["days_employed_sentinel"]
    unknown_gender = config["quality"]["gender_unknown_value"]
    employed_touched = scalar(con, f"SELECT count(*) FROM application_train WHERE DAYS_EMPLOYED = {sentinel}")
    gender_touched = scalar(con, f"SELECT count(*) FROM application_train WHERE CODE_GENDER = '{unknown_gender}'")
    employed_nulls = scalar(con, "SELECT count(*) FROM application_clean WHERE DAYS_EMPLOYED IS NULL") - scalar(
        con, "SELECT count(*) FROM application_train WHERE DAYS_EMPLOYED IS NULL")
    gender_nulls = scalar(con, "SELECT count(*) FROM application_clean WHERE CODE_GENDER IS NULL") - scalar(
        con, "SELECT count(*) FROM application_train WHERE CODE_GENDER IS NULL")
    flagged = scalar(con, "SELECT sum(DAYS_EMPLOYED_ANOMALY) FROM application_clean")
    if employed_nulls != employed_touched or gender_nulls != gender_touched or flagged != employed_touched:
        raise RuntimeError("cleaning view does not match the expected transformation counts")
    applied = [
        {"table": "application_train", "step": f"DAYS_EMPLOYED = {sentinel} set to null", "rows_touched": employed_touched},
        {"table": "application_train", "step": "DAYS_EMPLOYED_ANOMALY indicator added (1 where sentinel)", "rows_touched": flagged},
        {"table": "application_train", "step": f"CODE_GENDER = {unknown_gender} set to null", "rows_touched": gender_touched},
    ]
    filtered = leakage_filter_counts(con)
    for table, counts in filtered.items():
        applied.append({"table": table, "step": "rows with a positive day or month offset removed (leakage filter)", "rows_touched": counts["dropped_future_offset"]})
    applied.append({"table": "bureau_balance", "step": "rows whose bureau record was removed by the leakage filter",
                    "rows_touched": filtered["bureau_balance"]["dropped_parent_filtered"]})
    applied.append({"table": "bureau_balance", "step": "rows with no bureau record (orphans) removed by the join",
                    "rows_touched": filtered["bureau_balance"]["dropped_no_bureau_record"]})
    applied.append({"table": "previous_application", "step": "planned schedule dates after the application set to null",
                    "rows_touched": filtered["previous_application"]["masked_planned_dates"]})
    applied.append({"table": "previous_application", "step": f"schedule dates equal to {sentinel} set to null",
                    "rows_touched": filtered["previous_application"]["masked_sentinel_dates"]})
    return applied


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def render_markdown(report: dict[str, Any]) -> str:
    sections = ["# Data quality report", "Generated by `make quality` from the raw Home Credit tables."]
    overview = [[t, f"{p['rows']:,}", len(p["columns"]), ", ".join(p["duplicates"]["key"]), f"{p['duplicates']['duplicate_rows']:,}"]
                for t, p in report["tables"].items()]
    sections += ["## Tables", markdown_table(["table", "rows", "columns", "key checked", "duplicate key rows"], overview)]
    sections += ["## Transformations applied", markdown_table(
        ["table", "transformation", "rows touched"], [[t["table"], t["step"], f"{t['rows_touched']:,}"] for t in report["transformations"]])]
    anomaly_rows = [[a["table"], a["column"], a["description"], f"{a['rows']:,}", a["handling"]] for a in report["anomalies"]]
    sections += ["## Known anomalies", markdown_table(["table", "column", "anomaly", "rows", "handling"], anomaly_rows)]
    employed = report["anomalies"][0]
    sections.append(
        f"Applicants with the DAYS_EMPLOYED sentinel default at {percent(employed['default_rate_flagged'])} versus "
        f"{percent(employed['default_rate_other'])} for the rest, so the indicator carries signal and is kept as a feature."
    )
    for table, profile in report["tables"].items():
        constant = [c for c, p in profile["columns"].items() if p["constant"]]
        near_constant = [f"{c} ({percent(p['top_value_share'])})" for c, p in profile["columns"].items() if p["near_constant"]]
        missing = sorted(((c, p) for c, p in profile["columns"].items() if p["missing"] > 0), key=lambda item: -item[1]["missing_rate"])
        sections.append(f"## {table}")
        sections.append(f"Constant columns: {', '.join(constant) or 'none'}")
        sections.append(f"Near-constant columns (top value share at or above threshold): {', '.join(near_constant) or 'none'}")
        if missing:
            sections.append(f"Columns with missing values: {len(missing)}")
            sections.append("<details><summary>Missing rate per column</summary>\n\n" + markdown_table(
                ["column", "missing", "missing rate"], [[c, f"{p['missing']:,}", percent(p["missing_rate"])] for c, p in missing]) + "\n\n</details>")
        outlier_rows = sorted(profile["outliers"].items(), key=lambda item: -item[1]["outlier_rate"])
        if outlier_rows:
            sections.append("<details><summary>IQR outlier summary</summary>\n\n" + markdown_table(
                ["column", "q1", "q3", "lower fence", "upper fence", "min", "max", "outliers", "outlier rate"],
                [[c, f"{o['q1']:.4g}", f"{o['q3']:.4g}", f"{o['lower_fence']:.4g}", f"{o['upper_fence']:.4g}",
                  f"{o['min']:.4g}", f"{o['max']:.4g}", f"{o['outliers']:,}", percent(o["outlier_rate"])] for c, o in outlier_rows]
            ) + "\n\n</details>")
    return "\n\n".join(sections) + "\n"


def main() -> None:
    config = load_config()
    paths = get_paths()
    con = connect()
    prepare_views(con)
    tables = {table: profile_table(con, table, config) for table in config["tables"]}
    report = {"tables": tables, "anomalies": known_anomalies(con, config), "transformations": transformations(con, config)}
    write_json(paths.data_reports / "data_quality.json", report)
    (paths.data_reports / "data_quality.md").write_text(render_markdown(report))
    for item in report["transformations"]:
        print(f"{item['table']:<24}{item['step']:<64}{item['rows_touched']:>10,}")
    for table, profile in tables.items():
        flagged = sum(p["constant"] or p["near_constant"] for p in profile["columns"].values())
        print(f"{table:<24}rows {profile['rows']:>12,}  constant or near-constant {flagged:>3}  duplicate keys {profile['duplicates']['duplicate_rows']:,}")


if __name__ == "__main__":
    main()

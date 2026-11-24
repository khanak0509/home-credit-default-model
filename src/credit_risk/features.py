from pathlib import Path

import duckdb

from credit_risk.artifacts import markdown_table, write_json
from credit_risk.config import get_paths, load_config
from credit_risk.db import connect, read_sql, register_raw_views, run_sql_file

FEATURE_QUERIES = {
    "application": "10_application_features.sql",
    "installments": "20_installments_features.sql",
    "bureau": "30_bureau_features.sql",
    "bureau_balance": "40_bureau_balance_features.sql",
    "previous_application": "50_previous_application_features.sql",
    "pos_cash": "60_pos_cash_features.sql",
    "credit_card": "70_credit_card_features.sql",
}

FUTURE_OFFSET_CONDITIONS = {
    "installments_payments": "DAYS_INSTALMENT > 0 OR DAYS_ENTRY_PAYMENT > 0",
    "bureau": "DAYS_CREDIT > 0 OR DAYS_ENDDATE_FACT > 0 OR DAYS_CREDIT_UPDATE > 0",
    "bureau_balance": "MONTHS_BALANCE > 0",
    "previous_application": "DAYS_DECISION > 0",
    "pos_cash_balance": "MONTHS_BALANCE > 0",
    "credit_card_balance": "MONTHS_BALANCE > 0",
}

VALID_VIEWS = {
    "installments_payments": "installments_valid",
    "bureau": "bureau_valid",
    "bureau_balance": "bureau_balance_valid",
    "previous_application": "previous_application_valid",
    "pos_cash_balance": "pos_cash_valid",
    "credit_card_balance": "credit_card_valid",
}

PREVIOUS_DATE_COLUMNS = ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION"]

KEY = "SK_ID_CURR"


def cleaning_params() -> dict[str, object]:
    quality = load_config()["quality"]
    return {
        "days_employed_sentinel": quality["days_employed_sentinel"],
        "gender_unknown_value": quality["gender_unknown_value"],
    }


def prepare_views(con: duckdb.DuckDBPyConnection) -> None:
    register_raw_views(con)
    run_sql_file(con, "01_application_clean.sql", cleaning_params())
    run_sql_file(con, "02_valid_views.sql")


def scalar(con: duckdb.DuckDBPyConnection, query: str) -> int:
    return con.execute(query).fetchone()[0]


def leakage_filter_counts(con: duckdb.DuckDBPyConnection) -> dict[str, dict[str, int]]:
    counts = {}
    for table, condition in FUTURE_OFFSET_CONDITIONS.items():
        raw_rows = scalar(con, f"SELECT count(*) FROM {table}")
        kept_rows = scalar(con, f"SELECT count(*) FROM {VALID_VIEWS[table]}")
        counts[table] = {
            "raw_rows": raw_rows,
            "kept_rows": kept_rows,
            "dropped_rows": raw_rows - kept_rows,
            "dropped_future_offset": scalar(con, f"SELECT count(*) FROM {table} WHERE {condition}"),
        }
    balance = counts["bureau_balance"]
    balance["dropped_no_bureau_record"] = scalar(con, "SELECT count(*) FROM bureau_balance ANTI JOIN bureau USING (SK_ID_BUREAU)")
    balance["dropped_parent_filtered"] = scalar(
        con,
        "SELECT count(*) FROM bureau_balance SEMI JOIN bureau USING (SK_ID_BUREAU) "
        "WHERE SK_ID_BUREAU NOT IN (SELECT SK_ID_BUREAU FROM bureau_valid) AND MONTHS_BALANCE <= 0",
    )
    sentinel = load_config()["quality"]["days_employed_sentinel"]
    planned = " OR ".join(f"({c} > 0 AND {c} <> {sentinel})" for c in PREVIOUS_DATE_COLUMNS)
    unset = " OR ".join(f"{c} = {sentinel}" for c in PREVIOUS_DATE_COLUMNS)
    previous = counts["previous_application"]
    previous["masked_planned_dates"] = scalar(con, f"SELECT count(*) FROM previous_application WHERE DAYS_DECISION <= 0 AND ({planned})")
    previous["masked_sentinel_dates"] = scalar(con, f"SELECT count(*) FROM previous_application WHERE DAYS_DECISION <= 0 AND ({unset})")
    return counts


def build_group_tables(con: duckdb.DuckDBPyConnection) -> dict[str, list[str]]:
    columns_by_group = {}
    for group, query_file in FEATURE_QUERIES.items():
        con.execute(f"CREATE OR REPLACE TABLE {group}_features AS {read_sql(query_file)}")
        described = con.execute(f"DESCRIBE {group}_features").fetchall()
        columns_by_group[group] = [row[0] for row in described if row[0] != KEY]
    return columns_by_group


def categorical_levels(con: duckdb.DuckDBPyConnection) -> dict[str, list[str]]:
    described = con.execute("DESCRIBE application_features").fetchall()
    levels = {}
    for name, dtype, *_ in described:
        if dtype == "VARCHAR":
            values = con.execute(
                f"SELECT DISTINCT {name} FROM application_features WHERE {name} IS NOT NULL ORDER BY 1"
            ).fetchall()
            levels[name] = [value[0] for value in values]
    return levels


def write_feature_table(con: duckdb.DuckDBPyConnection, target: Path) -> None:
    joins = "\n".join(
        f"LEFT JOIN {group}_features USING ({KEY})" for group in FEATURE_QUERIES if group != "application"
    )
    con.execute(
        f"""
        COPY (
            SELECT application_clean.TARGET, application_features.*,
                {", ".join(f"{group}_features.* EXCLUDE ({KEY})" for group in FEATURE_QUERIES if group != "application")}
            FROM application_features
            JOIN application_clean USING ({KEY})
            {joins}
            ORDER BY {KEY}
        ) TO '{target}' (FORMAT parquet, COMPRESSION zstd)
        """
    )


def write_audit_attributes(con: duckdb.DuckDBPyConnection, target: Path) -> None:
    con.execute(
        f"""
        COPY (
            SELECT {KEY}, TARGET, CODE_GENDER, -DAYS_BIRTH / 365.25 AS AGE_YEARS
            FROM application_clean
            ORDER BY {KEY}
        ) TO '{target}' (FORMAT parquet)
        """
    )


def build_manifest(columns_by_group: dict[str, list[str]], levels: dict[str, list[str]]) -> dict[str, object]:
    config = load_config()
    groups = {feature: group for group, columns in columns_by_group.items() for feature in columns}
    return {
        "key": KEY,
        "target": "TARGET",
        "excluded": ["CODE_GENDER", "TARGET"],
        "groups": groups,
        "group_order": list(FEATURE_QUERIES),
        "categorical": levels,
        "external_score": [name for name in config["external_score_features"] if name in groups],
    }


def write_build_report(counts: dict[str, dict[str, int]], columns_by_group: dict[str, list[str]], rows: int) -> None:
    reports = get_paths().data_reports
    write_json(reports / "feature_build.json", {"leakage_filter": counts, "feature_counts": {g: len(c) for g, c in columns_by_group.items()}, "rows": rows})
    filter_rows = [[table, f"{c['raw_rows']:,}", f"{c['dropped_future_offset']:,}", f"{c.get('dropped_parent_filtered', 0):,}",
                    f"{c.get('dropped_no_bureau_record', 0):,}", f"{c['kept_rows']:,}"] for table, c in counts.items()]
    group_rows = [[group, len(columns)] for group, columns in columns_by_group.items()]
    text = "\n\n".join([
        "# Feature build",
        "## Leakage filter",
        "Behavioral rows with a positive day or month offset (after the application date) are removed before any aggregation. "
        "Bureau balance rows are also removed when their parent bureau record was removed by that filter, or when no bureau record exists "
        "for them at all (an orphan, not a leakage case).",
        markdown_table(["table", "raw rows", "future-dated rows", "parent record filtered", "no bureau record", "kept rows"], filter_rows),
        "Previous applications keep their row when the decision date is on or before the application, but the five schedule dates "
        f"({', '.join(PREVIOUS_DATE_COLUMNS)}) are set to null when positive: "
        f"{counts['previous_application']['masked_planned_dates']:,} rows carry a planned date after the application and "
        f"{counts['previous_application']['masked_sentinel_dates']:,} carry the {load_config()['quality']['days_employed_sentinel']} placeholder for an event that never happened.",
        "## Feature groups",
        markdown_table(["group", "features"], group_rows),
        f"Output: one row per SK_ID_CURR, {rows:,} rows.",
    ])
    (reports / "feature_build.md").write_text(text + "\n")


def main() -> None:
    paths = get_paths()
    con = connect()
    prepare_views(con)
    counts = leakage_filter_counts(con)
    columns_by_group = build_group_tables(con)
    levels = categorical_levels(con)
    features_path = paths.processed / "features.parquet"
    write_feature_table(con, features_path)
    write_audit_attributes(con, paths.processed / "audit_attributes.parquet")
    manifest = build_manifest(columns_by_group, levels)
    write_json(paths.processed / "features_manifest.json", manifest)
    write_json(paths.data_reports / "features_manifest.json", manifest)
    rows = con.execute(f"SELECT count(*), count(DISTINCT {KEY}) FROM read_parquet('{features_path}')").fetchone()
    if rows[0] != rows[1]:
        raise RuntimeError("features.parquet is not unique on SK_ID_CURR")
    write_build_report(counts, columns_by_group, rows[0])
    for table, count in counts.items():
        print(f"{table:<24}future-dated {count['dropped_future_offset']:>6,}  total dropped {count['dropped_rows']:>10,} of {count['raw_rows']:>12,}")
    for group, columns in columns_by_group.items():
        print(f"{group:<24}{len(columns):>4} features")
    print(f"features.parquet rows {rows[0]:,}")


if __name__ == "__main__":
    main()

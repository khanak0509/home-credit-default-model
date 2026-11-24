from pathlib import Path
from string import Template
from typing import Any

import duckdb

from credit_risk.config import get_paths, load_config


def connect(database: str = ":memory:") -> duckdb.DuckDBPyConnection:
    config = load_config()
    settings = config["duckdb"]
    temp_dir = get_paths().processed.parent.parent / settings["temp_directory"]
    temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database)
    con.execute(f"SET memory_limit = '{settings['memory_limit']}'")
    con.execute(f"SET threads = {int(settings['threads'])}")
    con.execute(f"SET temp_directory = '{temp_dir}'")
    con.execute("SET preserve_insertion_order = false")
    return con


def raw_csv_path(table: str) -> Path:
    config = load_config()
    return get_paths().raw / config["tables"][table]


def staged_parquet_path(table: str) -> Path:
    return get_paths().staged / f"{table}.parquet"


def register_raw_views(con: duckdb.DuckDBPyConnection, prefer_staged: bool = True) -> dict[str, str]:
    sources = {}
    for table in load_config()["tables"]:
        staged = staged_parquet_path(table)
        if prefer_staged and staged.exists():
            con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{staged}')")
            sources[table] = str(staged)
            continue
        csv_path = raw_csv_path(table)
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} is missing, run `make download`")
        con.execute(
            f"CREATE OR REPLACE VIEW {table} AS "
            f"SELECT * FROM read_csv('{csv_path}', header = true, sample_size = -1)"
        )
        sources[table] = str(csv_path)
    return sources


def read_sql(name: str, params: dict[str, Any] | None = None) -> str:
    sql_text = (get_paths().sql / name).read_text()
    return Template(sql_text).substitute(params or {})


def run_sql_file(con: duckdb.DuckDBPyConnection, name: str, params: dict[str, Any] | None = None) -> None:
    for statement in split_statements(read_sql(name, params)):
        con.execute(statement)


def split_statements(sql_text: str) -> list[str]:
    return [statement.strip() for statement in sql_text.split(";") if statement.strip()]

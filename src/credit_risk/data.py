from credit_risk.artifacts import write_json
from credit_risk.config import get_paths, load_config
from credit_risk.db import connect, raw_csv_path, register_raw_views, staged_parquet_path


def stage_tables() -> dict[str, int]:
    con = connect()
    register_raw_views(con, prefer_staged=False)
    row_counts = {}
    for table in load_config()["tables"]:
        target = staged_parquet_path(table)
        if not target.exists() or target.stat().st_mtime < raw_csv_path(table).stat().st_mtime:
            con.execute(f"COPY (SELECT * FROM {table}) TO '{target}' (FORMAT parquet, COMPRESSION zstd)")
        row_counts[table] = con.execute(f"SELECT count(*) FROM read_parquet('{target}')").fetchone()[0]
    return row_counts


def target_rate() -> tuple[int, float]:
    con = connect()
    register_raw_views(con)
    rows, rate = con.execute("SELECT count(*), avg(TARGET) FROM application_train").fetchone()
    return rows, rate


def main() -> None:
    row_counts = stage_tables()
    rows, rate = target_rate()
    write_json(get_paths().data_reports / "data_summary.json", {"row_counts": row_counts, "target_rate": rate, "applications": rows})
    for table, count in row_counts.items():
        print(f"{table:<24}{count:>12,}")
    print(f"target rate {rate:.4f} ({rows:,} applications)")


if __name__ == "__main__":
    main()

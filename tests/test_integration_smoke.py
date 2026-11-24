import pytest

from credit_risk.db import connect, raw_csv_path, register_raw_views


@pytest.fixture(scope="module")
def con():
    connection = connect()
    register_raw_views(connection)
    return connection


def test_raw_files_exist(config):
    missing = [table for table in config["tables"] if not raw_csv_path(table).exists()]
    assert not missing, f"missing raw files: {missing}, run `make download`"


@pytest.mark.parametrize("table", ["application_train", "bureau", "bureau_balance", "installments_payments"])
def test_row_counts_positive(con, table):
    assert con.execute(f"SELECT count(*) FROM {table}").fetchone()[0] > 0


def test_target_is_binary_with_minority_default(con):
    values, rate = con.execute("SELECT list(DISTINCT TARGET ORDER BY TARGET), avg(TARGET) FROM application_train").fetchone()
    assert values == [0, 1]
    assert 0 < rate < 0.5

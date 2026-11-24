import polars as pl
from sklearn.model_selection import StratifiedKFold, train_test_split

from credit_risk.artifacts import write_json
from credit_risk.config import get_paths, load_config
from credit_risk.db import connect, register_raw_views


def build_split() -> pl.DataFrame:
    config = load_config()
    con = connect()
    register_raw_views(con)
    labels = con.sql("SELECT SK_ID_CURR, TARGET FROM application_train ORDER BY SK_ID_CURR").pl()
    ids = labels["SK_ID_CURR"].to_numpy()
    target = labels["TARGET"].to_numpy()
    train_ids, holdout_ids, train_target, _ = train_test_split(
        ids, target, test_size=config["split"]["holdout_share"], stratify=target, random_state=config["seed"]
    )
    folds = StratifiedKFold(n_splits=config["split"]["n_folds"], shuffle=True, random_state=config["seed"])
    fold_by_id = {int(holdout_id): -1 for holdout_id in holdout_ids}
    for fold, (_, valid_index) in enumerate(folds.split(train_ids, train_target)):
        for row in valid_index:
            fold_by_id[int(train_ids[row])] = fold
    split = labels.with_columns(pl.col("SK_ID_CURR").replace_strict(fold_by_id, return_dtype=pl.Int8).alias("fold"))
    return split.with_columns((pl.col("fold") == -1).alias("is_holdout")).select("SK_ID_CURR", "fold", "is_holdout")


def main() -> None:
    paths = get_paths()
    split = build_split()
    split.write_parquet(paths.processed / "split.parquet")
    summary = split.group_by("fold").len().sort("fold")
    write_json(paths.data_reports / "split.json", {str(row["fold"]): row["len"] for row in summary.iter_rows(named=True)})
    print(f"holdout {split['is_holdout'].sum():,}  train {(~split['is_holdout']).sum():,}")


if __name__ == "__main__":
    main()

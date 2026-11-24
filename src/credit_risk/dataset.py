from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import polars as pl

from credit_risk.artifacts import read_json, require
from credit_risk.config import get_paths


@dataclass
class Dataset:
    X_train: pd.DataFrame
    y_train: np.ndarray
    folds: np.ndarray
    X_holdout: pd.DataFrame
    y_holdout: np.ndarray
    ids_train: np.ndarray
    ids_holdout: np.ndarray
    categorical: list[str] = field(default_factory=list)

    @property
    def features(self) -> list[str]:
        return list(self.X_train.columns)

    def fold_indices(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(np.flatnonzero(self.folds != k), np.flatnonzero(self.folds == k)) for k in np.unique(self.folds)]


def load_manifest() -> dict:
    return read_json(get_paths().processed / "features_manifest.json", "features")


def select_features(manifest: dict, groups: list[str] | None = None, drop: list[str] | None = None) -> list[str]:
    allowed = set(groups) if groups else set(manifest["group_order"])
    dropped = set(drop or [])
    return [name for name, group in manifest["groups"].items() if group in allowed and name not in dropped]


def apply_categories(frame: pd.DataFrame, levels: dict[str, list[str]]) -> pd.DataFrame:
    for column, values in levels.items():
        if column in frame.columns:
            frame[column] = pd.Categorical(frame[column], categories=values)
    return frame


def load_feature_frame(columns: list[str], ids: list[int] | None = None) -> pl.DataFrame:
    paths = get_paths()
    frame = pl.scan_parquet(require(paths.processed / "features.parquet", "features"))
    if ids is not None:
        frame = frame.filter(pl.col("SK_ID_CURR").is_in(ids))
    return frame.select(["SK_ID_CURR", "TARGET", *columns]).collect()


def to_model_frame(frame: pl.DataFrame, columns: list[str], manifest: dict) -> pd.DataFrame:
    pandas_frame = frame.select(columns).to_pandas()
    numeric = [c for c in columns if c not in manifest["categorical"]]
    pandas_frame[numeric] = pandas_frame[numeric].astype("float32")
    return apply_categories(pandas_frame, manifest["categorical"])


def load_dataset(groups: list[str] | None = None, drop: list[str] | None = None) -> Dataset:
    paths = get_paths()
    manifest = load_manifest()
    columns = select_features(manifest, groups, drop)
    split = pl.read_parquet(require(paths.processed / "split.parquet", "features"))
    frame = load_feature_frame(columns).join(split, on="SK_ID_CURR", how="inner").sort("SK_ID_CURR")
    train = frame.filter(~pl.col("is_holdout"))
    holdout = frame.filter(pl.col("is_holdout"))
    return Dataset(
        X_train=to_model_frame(train, columns, manifest),
        y_train=train["TARGET"].to_numpy().astype(np.int8),
        folds=train["fold"].to_numpy(),
        X_holdout=to_model_frame(holdout, columns, manifest),
        y_holdout=holdout["TARGET"].to_numpy().astype(np.int8),
        ids_train=train["SK_ID_CURR"].to_numpy(),
        ids_holdout=holdout["SK_ID_CURR"].to_numpy(),
        categorical=[c for c in columns if c in manifest["categorical"]],
    )

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

from credit_risk.dataset import Dataset
from credit_risk.metrics import fast_auc

WeightFn = Callable[[np.ndarray], np.ndarray]

warnings.filterwarnings("ignore", message="Overriding the parameters from Reference Dataset")


def sigmoid(margin: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-margin))


@dataclass
class FoldEnsemble:
    kind: str
    features: list[str]
    params: dict[str, Any]
    models: list[Any] = field(default_factory=list)
    best_iterations: list[int] = field(default_factory=list)

    def fold_margin(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        X = X[self.features]
        if self.kind == "lightgbm":
            return model.predict(X, raw_score=True)
        return model.predict(xgb.DMatrix(X, enable_categorical=True), output_margin=True, iteration_range=(0, model.best_iteration + 1))

    def predict_margin(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([self.fold_margin(model, X) for model in self.models], axis=0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return sigmoid(self.predict_margin(X))


def lightgbm_base_params(seed: int, n_jobs: int) -> dict[str, Any]:
    return {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": seed,
        "num_threads": n_jobs,
        "deterministic": True,
        "force_col_wise": True,
        "bagging_freq": 1,
    }


def xgboost_base_params(seed: int, n_jobs: int) -> dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "seed": seed,
        "nthread": n_jobs,
        "max_cat_to_onehot": 1,
        "verbosity": 0,
    }


def lightgbm_search_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 400, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.9),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 30.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "cat_smooth": trial.suggest_float("cat_smooth", 1.0, 50.0, log=True),
    }


def xgboost_search_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "eta": trial.suggest_float("eta", 0.03, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 100.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.9),
        "lambda": trial.suggest_float("lambda", 1e-3, 30.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }


SEARCH_SPACES = {"lightgbm": lightgbm_search_space, "xgboost": xgboost_search_space}


class FoldData:
    def __init__(self, kind: str, data: Dataset, weight_fn: WeightFn | None = None):
        self.kind = kind
        self.folds = data.fold_indices()
        self.y = data.y_train
        self.cache: list[tuple[Any, Any]] = []
        for train_index, valid_index in self.folds:
            X_fit, X_valid = data.X_train.iloc[train_index], data.X_train.iloc[valid_index]
            y_fit, y_valid = data.y_train[train_index], data.y_train[valid_index]
            weights = weight_fn(y_fit) if weight_fn else None
            if kind == "lightgbm":
                fit_set = lgb.Dataset(X_fit, y_fit, weight=weights, params={"feature_pre_filter": False, "verbosity": -1}, free_raw_data=False)
                valid_set = lgb.Dataset(X_valid, y_valid, reference=fit_set, free_raw_data=False)
            else:
                fit_set = xgb.QuantileDMatrix(X_fit, y_fit, weight=weights, enable_categorical=True)
                valid_set = xgb.QuantileDMatrix(X_valid, y_valid, ref=fit_set, enable_categorical=True)
            self.cache.append((fit_set, valid_set))


def train_fold(kind: str, params: dict[str, Any], fit_set: Any, valid_set: Any, max_rounds: int, patience: int) -> tuple[Any, int]:
    if kind == "lightgbm":
        booster = lgb.train(params, fit_set, num_boost_round=max_rounds, valid_sets=[valid_set],
                            callbacks=[lgb.early_stopping(patience, verbose=False)])
        return booster, booster.best_iteration
    booster = xgb.train(params, fit_set, num_boost_round=max_rounds, evals=[(valid_set, "valid")],
                        early_stopping_rounds=patience, verbose_eval=False)
    return booster, booster.best_iteration + 1


def cross_validate(kind: str, params: dict[str, Any], data: Dataset, max_rounds: int, patience: int,
                   weight_fn: WeightFn | None = None, fold_data: FoldData | None = None) -> tuple[np.ndarray, FoldEnsemble, list[float]]:
    fold_data = fold_data or FoldData(kind, data, weight_fn)
    ensemble = FoldEnsemble(kind=kind, features=data.features, params=params)
    oof = np.zeros(data.y_train.size)
    fold_aucs = []
    for (train_index, valid_index), (fit_set, valid_set) in zip(fold_data.folds, fold_data.cache):
        booster, best_iteration = train_fold(kind, params, fit_set, valid_set, max_rounds, patience)
        ensemble.models.append(booster)
        ensemble.best_iterations.append(best_iteration)
        oof[valid_index] = sigmoid(ensemble.fold_margin(booster, data.X_train.iloc[valid_index]))
        fold_aucs.append(fast_auc(data.y_train[valid_index], oof[valid_index]))
    return oof, ensemble, fold_aucs


def tune(kind: str, data: Dataset, base_params: dict[str, Any], n_trials: int, max_rounds: int, patience: int, seed: int) -> dict[str, Any]:
    fold_data = FoldData(kind, data)
    space = SEARCH_SPACES[kind]

    def objective(trial: optuna.Trial) -> float:
        params = {**base_params, **space(trial)}
        aucs = []
        for step, ((_, valid_index), (fit_set, valid_set)) in enumerate(zip(fold_data.folds, fold_data.cache)):
            booster, best_iteration = train_fold(kind, params, fit_set, valid_set, max_rounds, patience)
            if kind == "lightgbm":
                auc = booster.best_score["valid_0"]["auc"]
            else:
                auc = float(booster.best_score)
            aucs.append(auc)
            trial.report(float(np.mean(aucs)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(aucs))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
    )
    started = time.time()
    study.optimize(objective, n_trials=n_trials, callbacks=[progress_callback(kind, n_trials)])
    trials = [
        {"number": t.number, "state": t.state.name, "value": t.value, "params": t.params}
        for t in study.trials
    ]
    return {
        "best_value": study.best_value,
        "best_params": {**base_params, **study.best_params},
        "n_trials": n_trials,
        "n_pruned": sum(t["state"] == "PRUNED" for t in trials),
        "seconds": time.time() - started,
        "trials": trials,
    }


def progress_callback(kind: str, n_trials: int) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    def report(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        value = f"{trial.value:.5f}" if trial.value is not None else "pruned"
        print(f"{kind} trial {trial.number + 1}/{n_trials} {value} best {study.best_value:.5f}", flush=True)

    return report

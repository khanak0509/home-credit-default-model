import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_ENV = "CREDIT_RISK_PROFILE"


@dataclass(frozen=True)
class Paths:
    raw: Path
    staged: Path
    processed: Path
    models: Path
    reports: Path
    data_reports: Path
    figures: Path
    sql: Path

    def ensure(self) -> None:
        for path in (self.staged, self.processed, self.models, self.reports, self.figures):
            path.mkdir(parents=True, exist_ok=True)


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        nested = isinstance(value, dict) and isinstance(base.get(key), dict)
        merged[key] = merge(base[key], value) if nested else value
    return merged


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    profiles = config.pop("profiles", {})
    profile = os.environ.get(PROFILE_ENV) or "full"
    if profile != "full":
        if profile not in profiles:
            raise ValueError(f"unknown {PROFILE_ENV}={profile}, expected full or one of {sorted(profiles)}")
        config = merge(config, profiles[profile])
    config["profile"] = profile
    return config


def get_paths(config: dict[str, Any] | None = None) -> Paths:
    config = config or load_config()
    resolved = {key: PROJECT_ROOT / value for key, value in config["paths"].items()}
    paths = Paths(**resolved)
    paths.ensure()
    return paths

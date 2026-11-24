import json
from pathlib import Path
from typing import Any

import numpy as np


class MissingArtifactError(FileNotFoundError):
    pass


def require(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise MissingArtifactError(f"{path} not found, run `make {produced_by}` first")
    return path


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.ndarray):
        return _to_builtin(value.tolist())
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_builtin(payload), indent=2) + "\n")


def read_json(path: Path, produced_by: str) -> Any:
    return json.loads(require(path, produced_by).read_text())


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def signed(value: float, digits: int = 4) -> str:
    rounded = round(value, digits) + 0.0
    return f"{rounded:+.{digits}f}"

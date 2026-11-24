import shutil
import subprocess
import zipfile
from pathlib import Path

from credit_risk.config import get_paths, load_config


def _unzip_all(folder: Path) -> None:
    for archive in folder.glob("*.zip"):
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(folder)
        archive.unlink()


def _flatten(folder: Path, file_name: str) -> None:
    for candidate in folder.rglob(file_name):
        target = folder / file_name
        if candidate != target:
            shutil.move(str(candidate), target)


def _kaggle(args: list[str]) -> bool:
    result = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    return result.returncode == 0


def download_table(file_name: str, raw_dir: Path) -> None:
    source = load_config()["download"]
    if _kaggle(["competitions", "download", source["competition"], "-f", file_name, "-p", str(raw_dir), "-q"]):
        return
    for remote_name in (file_name, f"{source['competition']}/{file_name}"):
        if _kaggle(["datasets", "download", source["mirror_dataset"], "-f", remote_name, "-p", str(raw_dir), "-q"]):
            return
    raise RuntimeError(f"could not download {file_name}, accept the competition rules on kaggle.com first")


def main() -> None:
    raw_dir = get_paths().raw
    raw_dir.mkdir(parents=True, exist_ok=True)
    for file_name in load_config()["tables"].values():
        if (raw_dir / file_name).exists():
            continue
        download_table(file_name, raw_dir)
        _unzip_all(raw_dir)
        _flatten(raw_dir, file_name)
        print(f"downloaded {file_name}")
    for leftover in raw_dir.iterdir():
        if leftover.is_dir():
            shutil.rmtree(leftover)


if __name__ == "__main__":
    main()

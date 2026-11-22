"""Project paths and small file helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def project_root() -> Path:
    """Return the Synthetic_Data_ML directory."""
    return Path(__file__).resolve().parents[2]


def data_raw_dir() -> Path:
    return project_root() / "data" / "raw"


def data_processed_dir() -> Path:
    return project_root() / "data" / "processed"


def models_dir() -> Path:
    return project_root() / "models"


def results_dir() -> Path:
    return project_root() / "results"


def tables_dir() -> Path:
    return results_dir() / "tables"


def figures_dir() -> Path:
    return results_dir() / "figures"


def reports_dir() -> Path:
    return project_root() / "reports"


def config_path() -> Path:
    return project_root() / "configs" / "phase1.json"


def resolve_zip_path() -> Path:
    """Locate the original Adult ZIP without modifying it.

    Resolution order:
    1. ``ADULT_DATASET_ZIP`` environment variable
    2. ``../Adult Dataset.zip`` relative to the project root
    """
    env = os.environ.get("ADULT_DATASET_ZIP")
    if env:
        return Path(env).expanduser().resolve()
    return (project_root().parent / "Adult Dataset.zip").resolve()


def ensure_project_dirs() -> None:
    for path in (
        data_raw_dir(),
        data_processed_dir(),
        project_root() / "data" / "synthetic",
        models_dir(),
        tables_dir(),
        figures_dir(),
        reports_dir(),
    ):
        path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict:
    with config_path().open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value):
        if hasattr(value, "item"):
            return value.item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, default=convert)
        handle.write("\n")

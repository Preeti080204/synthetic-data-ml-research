"""Load the UCI Adult Census Income files without altering the raw release."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from data.schema import (
    CATEGORICAL_FEATURES,
    EXPECTED_RAW_SHA256,
    EXPECTED_ROW_COUNTS,
    FEATURE_COLUMNS,
    METADATA_COLUMNS,
    NEGATIVE_CLASS,
    NUMERIC_FEATURES,
    POSITIVE_CLASS,
    TARGET_COLUMN,
)
from utils.paths import data_raw_dir, resolve_zip_path, sha256_file

RAW_REQUIRED = ("adult.data", "adult.test", "adult.names")
VALID_LABELS = {NEGATIVE_CLASS, POSITIVE_CLASS}


@dataclass
class LoadResult:
    """Combined Adult table plus loading audit information."""

    frame: pd.DataFrame
    metadata: dict = field(default_factory=dict)


def ensure_raw_dataset() -> Path:
    """Return the raw directory, extracting the ZIP only if files are absent.

    Extraction reads the ZIP and writes copies under ``data/raw``. The original
    archive is opened read-only and is never rewritten.
    """
    raw_dir = data_raw_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)
    if all((raw_dir / name).is_file() for name in RAW_REQUIRED):
        return raw_dir

    zip_path = resolve_zip_path()
    if not zip_path.is_file():
        raise FileNotFoundError(
            "Adult raw files are missing and the ZIP was not found. Place "
            "adult.data, adult.test, and adult.names in data/raw/, or set "
            "ADULT_DATASET_ZIP, or keep 'Adult Dataset.zip' in the parent "
            f"directory. Looked for: {zip_path}"
        )
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(raw_dir)
    missing = [name for name in RAW_REQUIRED if not (raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"ZIP did not contain required files: {missing}")
    return raw_dir


def _normalize_label(value: str) -> str:
    return str(value).strip().removesuffix(".").strip()


def _read_adult_file(path: Path, source_file: str) -> tuple[pd.DataFrame, dict]:
    """Parse one Adult file, skipping blank lines and the test-file banner."""
    records: list[tuple[int, list[str]]] = []
    banner_lines = 0
    blank_lines = 0
    trailing_period_labels = 0

    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                blank_lines += 1
                continue
            if text.startswith("|"):
                banner_lines += 1
                continue
            parts = [part.strip() for part in text.split(",")]
            if len(parts) != 15:
                raise ValueError(
                    f"{source_file}:{line_number} has {len(parts)} comma-separated "
                    "fields; expected 15."
                )
            label = parts[-1]
            if label.endswith("."):
                trailing_period_labels += 1
            parts[-1] = _normalize_label(label)
            records.append((line_number, parts))

    if not records:
        raise ValueError(f"{source_file} did not contain any data rows.")

    frame = pd.DataFrame(
        [parts for _, parts in records],
        columns=FEATURE_COLUMNS + [TARGET_COLUMN],
    )
    frame.insert(0, "source_line", [line_number for line_number, _ in records])
    frame.insert(0, "source_file", source_file)
    frame.insert(
        2,
        "source_row_id",
        frame["source_file"] + ":" + frame["source_line"].astype(str),
    )

    audit = {
        "source_file": source_file,
        "data_rows": int(len(frame)),
        "banner_lines": int(banner_lines),
        "blank_lines": int(blank_lines),
        "trailing_period_labels": int(trailing_period_labels),
        "sha256": sha256_file(path),
    }
    return frame, audit


def _coerce_types(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in NUMERIC_FEATURES:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="raise")

    for column in CATEGORICAL_FEATURES:
        values = cleaned[column].astype("object")
        values = values.where(~values.isin(["", "?"]), np.nan)
        cleaned[column] = values

    labels = cleaned[TARGET_COLUMN].map(_normalize_label)
    invalid = ~labels.isin(VALID_LABELS)
    if bool(invalid.any()):
        examples = labels.loc[invalid].unique().tolist()[:5]
        raise ValueError(f"Unrecognized income labels: {examples}")
    cleaned[TARGET_COLUMN] = labels
    return cleaned


def load_adult_combined(raw_dir: Path | None = None) -> LoadResult:
    """Load, clean, and stack ``adult.data`` and ``adult.test``.

    Cleaning strips whitespace, maps ``?`` to missing values, and removes the
    trailing period that the UCI test file appends to every income label.
    Rows are not deleted. The returned row order is ``adult.data`` followed by
    ``adult.test``, each in original file order.
    """
    raw_dir = ensure_raw_dataset() if raw_dir is None else raw_dir
    frames = []
    file_audits = []
    for source_file in ("adult.data", "adult.test"):
        frame, audit = _read_adult_file(raw_dir / source_file, source_file)
        frames.append(frame)
        file_audits.append(audit)

    combined = pd.concat(frames, ignore_index=True)
    combined = _coerce_types(combined)
    combined = combined.loc[:, METADATA_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]]

    if combined["source_row_id"].duplicated().any():
        raise ValueError("source_row_id values are not unique.")

    for audit in file_audits:
        expected = EXPECTED_ROW_COUNTS[audit["source_file"]]
        if audit["data_rows"] != expected:
            raise ValueError(
                f"{audit['source_file']} produced {audit['data_rows']} rows; "
                f"expected {expected}."
            )
        known = EXPECTED_RAW_SHA256.get(audit["source_file"])
        audit["sha256_matches_project_release"] = audit["sha256"] == known

    if len(combined) != EXPECTED_ROW_COUNTS["combined"]:
        raise ValueError(
            f"Combined row count is {len(combined)}; "
            f"expected {EXPECTED_ROW_COUNTS['combined']}."
        )

    metadata = {
        "files": file_audits,
        "n_rows": int(len(combined)),
        "n_features": len(FEATURE_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "positive_class": POSITIVE_CLASS,
        "negative_class": NEGATIVE_CLASS,
        "rows_removed": 0,
        "trailing_period_labels": int(
            sum(audit["trailing_period_labels"] for audit in file_audits)
        ),
    }
    return LoadResult(frame=combined, metadata=metadata)


def save_processed_dataset(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)

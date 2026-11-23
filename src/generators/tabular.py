"""Gaussian copula and CTGAN fits on real training rows only.

Completed runs are cached. Invalid target labels are counted and, when
necessary, replaced by extra documented sampling rounds. Rows are not dropped
without a log, and they are not duplicated to hide a shortfall.
"""

from __future__ import annotations

import inspect
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from data.schema import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NEGATIVE_CLASS,
    NUMERIC_FEATURES,
    POSITIVE_CLASS,
    TARGET_COLUMN,
)
from experiments.design import generator_config
from experiments.tracking import is_complete, read_json, utc_now, write_record
from utils.paths import project_root, sha256_file

TABLE_NAME = "adult"
VALID_LABELS = {POSITIVE_CLASS, NEGATIVE_CLASS}


def prepare_generator_frame(frame: pd.DataFrame, missing_level: str = "Missing") -> pd.DataFrame:
    """Copy model columns and encode categorical missingness as an explicit level.

    The source frame is not modified. Numerical missing values are left as
    missing because the training partition has none. Categorical ``NA`` becomes
    the same ``Missing`` level used by the Phase 1 preprocessor.
    """
    columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Generator frame is missing {missing}.")
    prepared = frame.loc[:, columns].copy()
    for column in CATEGORICAL_FEATURES:
        prepared[column] = prepared[column].where(prepared[column].notna(), missing_level)
        prepared[column] = prepared[column].astype(str)
    prepared[TARGET_COLUMN] = (
        prepared[TARGET_COLUMN].astype(str).str.strip().str.rstrip(".")
    )
    unexpected = set(prepared[TARGET_COLUMN].unique()) - VALID_LABELS
    if unexpected:
        raise ValueError(f"Real training target has unexpected labels: {unexpected}.")
    return prepared


def _seed_libraries(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _metadata(frame: pd.DataFrame):
    from sdv.metadata import Metadata

    metadata = Metadata.detect_from_dataframes({TABLE_NAME: frame}, infer_keys="primary_only")
    # fnlwgt is a survey weight. On a small subset it can be unique, and SDV
    # then treats it as a primary key. It is not an identifier.
    try:
        metadata.remove_primary_key(TABLE_NAME)
    except Exception:
        pass
    for column in NUMERIC_FEATURES:
        metadata.update_column(column_name=column, sdtype="numerical")
    for column in list(CATEGORICAL_FEATURES) + [TARGET_COLUMN]:
        metadata.update_column(column_name=column, sdtype="categorical")
    return metadata


def _filter_kwargs(callable_obj, values: dict) -> dict:
    signature = inspect.signature(callable_obj)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return dict(values)
    accepted = set(signature.parameters)
    return {key: value for key, value in values.items() if key in accepted}


def build_synthesizer(kind: str, metadata, config: dict | None = None, overrides: dict | None = None):
    config = generator_config() if config is None else config
    overrides = overrides or {}
    if kind == "gaussian_copula":
        from sdv.single_table import GaussianCopulaSynthesizer

        settings = dict(config["gaussian_copula"])
        settings.pop("distribution_note", None)
        settings.update(overrides)
        kwargs = _filter_kwargs(GaussianCopulaSynthesizer.__init__, settings)
        return GaussianCopulaSynthesizer(metadata, **kwargs), kwargs
    if kind == "ctgan":
        from sdv.single_table import CTGANSynthesizer

        settings = dict(config["ctgan"])
        settings.pop("epoch_budget", None)
        for key in ("generator_dim", "discriminator_dim"):
            if key in settings:
                settings[key] = tuple(settings[key])
        settings.update(overrides)
        kwargs = _filter_kwargs(CTGANSynthesizer.__init__, settings)
        return CTGANSynthesizer(metadata, **kwargs), kwargs
    raise ValueError(f"Unknown generator {kind!r}.")


def _sample_until_valid(synthesizer, n_rows: int, seed: int, max_rounds: int) -> tuple[pd.DataFrame, list[dict], str]:
    pieces: list[pd.DataFrame] = []
    attempts: list[dict] = []
    remaining = int(n_rows)
    for round_index in range(1, max_rounds + 1):
        round_seed = seed if round_index == 1 else seed + round_index * 10007
        synthesizer._set_random_state(round_seed)
        sampled = synthesizer.sample(num_rows=remaining)
        if isinstance(sampled, dict):
            sampled = sampled[TABLE_NAME]
        labels = sampled[TARGET_COLUMN].astype(str).str.strip().str.rstrip(".")
        sampled = sampled.copy()
        sampled[TARGET_COLUMN] = labels
        valid = labels.isin(VALID_LABELS)
        attempts.append(
            {
                "round": round_index,
                "seed": int(round_seed),
                "sampled": int(len(sampled)),
                "invalid_target": int((~valid).sum()),
            }
        )
        if valid.any():
            pieces.append(sampled.loc[valid].copy())
        remaining = n_rows - sum(len(piece) for piece in pieces)
        if remaining <= 0:
            break
    if not pieces:
        empty = sampled.iloc[0:0].copy() if "sampled" in locals() else pd.DataFrame()
        return empty, attempts, "shortfall"
    combined = pd.concat(pieces, ignore_index=True).iloc[:n_rows].reset_index(drop=True)
    status = "completed" if len(combined) == n_rows else "shortfall"
    return combined, attempts, status


def _domain_report(real_frame: pd.DataFrame, synthetic: pd.DataFrame) -> dict:
    categorical = {}
    for column in CATEGORICAL_FEATURES:
        observed = set(real_frame[column].astype(str).unique())
        generated = synthetic[column].astype(str)
        unseen = sorted(set(generated.unique()) - observed)
        categorical[column] = {
            "unseen_levels": unseen,
            "unseen_rows": int(generated.isin(unseen).sum()) if unseen else 0,
            "missing_rows": int(synthetic[column].isna().sum()),
        }
    numerical = {}
    for column in NUMERIC_FEATURES:
        observed = pd.to_numeric(real_frame[column], errors="coerce")
        generated = pd.to_numeric(synthetic[column], errors="coerce")
        low = float(observed.min())
        high = float(observed.max())
        outside = int(((generated < low) | (generated > high)).sum())
        numerical[column] = {
            "training_min": low,
            "training_max": high,
            "outside_training_range": outside,
            "missing_rows": int(generated.isna().sum()),
        }
    return {"categorical": categorical, "numerical": numerical}


def _planned_hyperparameters(kind: str, config: dict, overrides: dict | None) -> dict:
    overrides = dict(overrides or {})
    if kind == "gaussian_copula":
        from sdv.single_table import GaussianCopulaSynthesizer

        settings = dict(config["gaussian_copula"])
        settings.pop("distribution_note", None)
        settings.update(overrides)
        return _jsonable(_filter_kwargs(GaussianCopulaSynthesizer.__init__, settings))
    if kind == "ctgan":
        from sdv.single_table import CTGANSynthesizer

        settings = dict(config["ctgan"])
        settings.pop("epoch_budget", None)
        for key in ("generator_dim", "discriminator_dim"):
            if key in settings:
                settings[key] = tuple(settings[key])
        settings.update(overrides)
        return _jsonable(_filter_kwargs(CTGANSynthesizer.__init__, settings))
    raise ValueError(f"Unknown generator {kind!r}.")


def _cache_request(prepared, kind, seed, n_rows, overrides, destination, data_path, meta_path, config) -> dict:
    from experiments.cache import file_sha256, frame_fingerprint

    generators = project_root() / "configs" / "generators.yaml"
    return {
        "generator": kind,
        "seed": int(seed),
        "n_rows": int(n_rows),
        "n_training_rows": int(len(prepared)),
        "hyperparameters": _planned_hyperparameters(kind, config, overrides),
        "library_versions": _library_versions(),
        "training_fingerprint": frame_fingerprint(prepared),
        "generators_yaml_sha256": file_sha256(generators),
        "synthetic_sha256": file_sha256(data_path) if data_path.is_file() else None,
        "generation_json_sha256": file_sha256(meta_path) if meta_path.is_file() else None,
        "upstream_synthetic_sha256": None,
    }


def generation_dir(root: Path, kind: str, seed: int) -> Path:
    return root / kind / f"seed_{seed}"


def generate_cached(
    real_frame: pd.DataFrame,
    kind: str,
    seed: int,
    root: Path,
    n_rows: int | None = None,
    overrides: dict | None = None,
) -> dict:
    """Fit on ``real_frame`` only and cache N synthetic rows.

    A completed metadata file with a matching row count skips the fit.
    """
    config = generator_config()
    destination = generation_dir(root, kind, seed)
    destination.mkdir(parents=True, exist_ok=True)
    meta_path = destination / "generation.json"
    data_path = destination / "synthetic.csv"
    prepared = prepare_generator_frame(real_frame, config.get("missing_level", "Missing"))
    n_rows = len(prepared) if n_rows is None else int(n_rows)
    cached = read_json(meta_path)
    request = _cache_request(
        prepared, kind, seed, n_rows, overrides, destination, data_path, meta_path, config
    )
    from experiments.cache import generator_cache_hit, load_ledger

    ledger = load_ledger() or {}
    relative = destination.relative_to(project_root()).as_posix()
    if generator_cache_hit(cached, request, (ledger.get("entries") or {}).get(relative)):
        cached["cache"] = "hit"
        cached["cache_validation"] = "matched stored identity"
        return cached

    warnings_seen: list[str] = []
    started = time.perf_counter()
    try:
        _seed_libraries(int(seed))
        metadata = _metadata(prepared)
        metadata_path = destination / "metadata.json"
        if metadata_path.exists():
            metadata_path.unlink()
        metadata.save_to_json(metadata_path)
        from sdv.metadata import Metadata

        metadata = Metadata.load_from_json(metadata_path)
        synthesizer, used_kwargs = build_synthesizer(kind, metadata, config, overrides)
        print(
            f"fitting {kind} seed={seed} epochs={used_kwargs.get('epochs')} "
            f"enable_gpu={used_kwargs.get('enable_gpu')} rows={len(prepared)}",
            flush=True,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            synthesizer.fit({TABLE_NAME: prepared})
        print(f"fit returned {kind} seed={seed}", flush=True)
        warnings_seen.extend(f"{item.category.__name__}: {item.message}" for item in caught)
        synthetic, attempts, sample_status = _sample_until_valid(
            synthesizer,
            n_rows,
            int(seed),
            int(config["invalid_target_policy"]["max_rounds"]),
        )
        invalid_total = int(sum(item["invalid_target"] for item in attempts))
        domains = _domain_report(prepared, synthetic) if len(synthetic) else {}
        model_path = destination / "synthesizer.pkl"
        synthesizer.save(str(model_path))
        loss_path = None
        loss_values = getattr(synthesizer, "loss_values", None)
        if isinstance(loss_values, pd.DataFrame) and len(loss_values):
            loss_path = destination / "loss_values.csv"
            loss_values.to_csv(loss_path, index=False)
        if sample_status == "completed":
            synthetic.to_csv(data_path, index=False)
        elapsed = time.perf_counter() - started
        payload = {
            "status": sample_status,
            "generator": kind,
            "seed": int(seed),
            "n_rows": int(len(synthetic)),
            "n_requested": int(n_rows),
            "n_training_rows": int(len(prepared)),
            "fit_seconds": elapsed,
            "timestamp": utc_now(),
            "hyperparameters": _jsonable(used_kwargs),
            "invalid_target_rounds": attempts,
            "invalid_target_rows": invalid_total,
            "invalid_target_policy": config["invalid_target_policy"]["name"],
            "rows_dropped_silently": False,
            "domain_checks": domains,
            "warnings": warnings_seen[:50],
            "paths": {
                "synthetic": str(data_path),
                "metadata": str(metadata_path),
                "synthesizer": str(model_path),
                "loss": str(loss_path) if loss_path else None,
            },
            "library_versions": _library_versions(),
            "accelerator": _accelerator_note(kind, used_kwargs),
            "cache": "miss",
            "cache_contract": 2,
            "training_fingerprint": request["training_fingerprint"],
            "generators_yaml_sha256": request["generators_yaml_sha256"],
            "upstream_synthetic_sha256": None,
        }
        if sample_status != "completed":
            payload["status"] = "failed"
            payload["error"] = (
                f"Generated {len(synthetic)} valid rows out of {n_rows}. "
                "No rows were duplicated to fill the shortfall."
            )
        write_record(meta_path, payload)
        return payload
    except Exception as exc:
        elapsed = time.perf_counter() - started
        payload = {
            "status": "failed",
            "generator": kind,
            "seed": int(seed),
            "n_requested": int(n_rows),
            "fit_seconds": elapsed,
            "timestamp": utc_now(),
            "error": f"{type(exc).__name__}: {exc}",
            "warnings": warnings_seen[:50],
            "cache": "miss",
        }
        write_record(meta_path, payload)
        return payload


def load_synthetic(root: Path, kind: str, seed: int) -> pd.DataFrame:
    path = generation_dir(root, kind, seed) / "synthetic.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing synthetic data: {path}")
    frame = pd.read_csv(path)
    if not is_complete(generation_dir(root, kind, seed) / "generation.json"):
        raise RuntimeError(f"Synthetic data at {path} is not a completed run.")
    return frame


def _accelerator_note(kind: str, kwargs: dict) -> str:
    if kind != "ctgan":
        return "cpu"
    import torch

    if kwargs.get("enable_gpu") and torch.backends.mps.is_available():
        return "mps"
    if kwargs.get("enable_gpu") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _library_versions() -> dict:
    import importlib.metadata

    versions = {}
    for package in ("sdv", "ctgan", "copulas", "rdt", "sdmetrics", "torch", "pandas"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def synthetic_file_hash(root: Path, kind: str, seed: int) -> str | None:
    path = generation_dir(root, kind, seed) / "synthetic.csv"
    if not path.is_file():
        return None
    return sha256_file(path)

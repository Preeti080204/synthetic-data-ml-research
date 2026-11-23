"""Cache identity checks.

Historical generator records did not store a training-data fingerprint.
Those runs stay on disk. A later call may reuse one only when the fields
that were stored still match and an audit ledger, computed afterwards,
confirms the file bytes and a recomputed training fingerprint. A new run
writes cache contract 2, which carries the fingerprints itself.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from utils.paths import project_root


def frame_fingerprint(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_fingerprint() -> str:
    """Hash tracked source files. This is not a claim about an earlier run."""
    root = project_root()
    digest = hashlib.sha256()
    roots = [root / "src", root / "scripts", root / "configs", root / "tests"]
    files = [root / "pyproject.toml"]
    for base in roots:
        if base.is_dir():
            files.extend(path for path in base.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    for path in sorted(files):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(file_sha256(path).encode())
    return digest.hexdigest()


def ledger_path() -> Path:
    return project_root() / "results" / "provenance" / "legacy_cache_ledger.json"


def load_ledger() -> dict | None:
    path = ledger_path()
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _same(left, right) -> bool:
    return json.dumps(left, sort_keys=True, default=str) == json.dumps(right, sort_keys=True, default=str)


def generator_cache_hit(cached: dict | None, request: dict, ledger_entry: dict | None) -> bool:
    """Return whether a saved generator record may be reused for this request."""
    if not cached or cached.get("status") != "completed":
        return False
    if int(cached.get("seed", -1)) != int(request["seed"]):
        return False
    if cached.get("generator") != request["generator"]:
        return False
    if int(cached.get("n_rows", -1)) != int(request["n_rows"]):
        return False
    if int(cached.get("n_training_rows", -1)) != int(request["n_training_rows"]):
        return False
    if not _same(cached.get("hyperparameters"), request["hyperparameters"]):
        return False
    if not _same(cached.get("library_versions"), request["library_versions"]):
        return False
    if int(cached.get("cache_contract", 0)) >= 2:
        return (
            cached.get("training_fingerprint") == request["training_fingerprint"]
            and cached.get("generators_yaml_sha256") == request["generators_yaml_sha256"]
            and cached.get("upstream_synthetic_sha256") == request.get("upstream_synthetic_sha256")
        )
    if ledger_entry is None:
        return False
    if ledger_entry.get("recorded_at_training_time"):
        return False
    return (
        ledger_entry.get("synthetic_sha256") == request["synthetic_sha256"]
        and ledger_entry.get("generation_json_sha256") == request["generation_json_sha256"]
        and ledger_entry.get("recomputed_training_fingerprint") == request["training_fingerprint"]
        and ledger_entry.get("note") == "recomputed_at_audit_not_stored_in_the_run"
    )


def classifier_cache_hit(cached: dict | None, context: dict, params: dict) -> bool:
    """Reuse a classifier record only when its stored identity matches."""
    if not cached or cached.get("status") != "completed":
        return False
    metrics = cached.get("metrics") or {}
    required = ("roc_auc", "f1", "precision", "recall", "balanced_accuracy")
    if any(metrics.get(name) is None for name in required):
        return False
    if str(cached.get("test_metrics_computed")).lower() in {"true", "1"}:
        return False
    if cached.get("scored_on") not in (None, "validation"):
        return False
    identity = (
        cached.get("experiment_id") == context.get("experiment_id")
        and cached.get("dataset_fingerprint") == context.get("dataset_fingerprint")
        and cached.get("split_fingerprint") == context.get("split_fingerprint")
        and cached.get("config_fingerprint") == context.get("config_fingerprint")
        and int(cached.get("seed", -1)) == int(context.get("seed", -2))
        and cached.get("classifier") == context.get("classifier")
        and int(cached.get("n_real", -1)) == int(context.get("n_real", -2))
        and int(cached.get("n_synthetic", -1)) == int(context.get("n_synthetic", -2))
        and int(cached.get("n_train", -1)) == int(context.get("n_real", 0)) + int(context.get("n_synthetic", 0))
        and _same(cached.get("hyperparameters"), params)
    )
    return identity

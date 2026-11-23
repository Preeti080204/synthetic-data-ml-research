"""Record the execution environment for the Phase 1 report."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from datetime import datetime, timezone

TRACKED_PACKAGES = [
    "pandas",
    "numpy",
    "scikit-learn",
    "xgboost",
    "matplotlib",
    "seaborn",
    "scipy",
    "joblib",
    "fpdf2",
]


def collect_environment() -> dict:
    versions = {}
    for package in TRACKED_PACKAGES:
        versions[package] = importlib.metadata.version(package)
    return {
        "python_version": platform.python_version(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": versions,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

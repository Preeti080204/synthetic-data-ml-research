"""Shared fixtures for the Adult Phase 1 tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.load_data import load_adult_combined
from data.split_data import make_splits
from utils.paths import load_config


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def loaded():
    return load_adult_combined(ROOT / "data" / "raw")


@pytest.fixture(scope="session")
def adult_df(loaded):
    return loaded.frame


@pytest.fixture(scope="session")
def splits(adult_df, config):
    return make_splits(adult_df, config)

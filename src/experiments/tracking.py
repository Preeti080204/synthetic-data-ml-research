"""JSON checkpoints for generator fits and classifier runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from utils.paths import write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def is_complete(path: Path) -> bool:
    payload = read_json(path)
    return bool(payload) and payload.get("status") == "completed"


def write_record(path: Path, payload: dict) -> None:
    payload = dict(payload)
    payload.setdefault("timestamp", utc_now())
    write_json(path, payload)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=_convert) + "\n")


def _convert(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

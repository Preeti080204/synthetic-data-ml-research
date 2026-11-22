"""Hash the completed validation outputs without rewriting them.

Training fingerprints in the ledger are recomputed at audit time. They were
not stored in the original generation records.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.cache import file_sha256, frame_fingerprint, ledger_path, source_tree_fingerprint
from experiments.data import load_modeling_frames, modeling_columns
from experiments.design import budget_count, stratified_sample
from experiments.tracking import utc_now
from generators.tabular import prepare_generator_frame
from utils.paths import project_root, write_json


def main() -> None:
    train, _validation = load_modeling_frames()
    root = project_root()
    entries = {}
    for meta in sorted((root / "data" / "synthetic").rglob("generation.json")):
        if "smoke" in meta.parts:
            continue
        payload = json.loads(meta.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            continue
        synthetic = meta.with_name("synthetic.csv")
        prepared = _training_frame(train, meta, int(payload["seed"]), int(payload["n_training_rows"]))
        relative = meta.parent.relative_to(root).as_posix()
        from experiments.design import generator_config
        from generators.tabular import _planned_hyperparameters

        planned = _planned_hyperparameters(payload["generator"], generator_config(), None)
        if planned != payload.get("hyperparameters"):
            raise RuntimeError(f"Current generator settings do not match the stored record at {meta}.")
        entries[relative] = {
            "note": "recomputed_at_audit_not_stored_in_the_run",
            "recorded_at_training_time": False,
            "synthetic_sha256": file_sha256(synthetic),
            "generation_json_sha256": file_sha256(meta),
            "recomputed_training_fingerprint": frame_fingerprint(prepared),
            "recorded_generator": payload.get("generator"),
            "recorded_seed": payload.get("seed"),
            "recorded_n_rows": payload.get("n_rows"),
            "recorded_n_training_rows": payload.get("n_training_rows"),
            "fields_present_in_generation_json": sorted(payload.keys()),
            "training_fingerprint_was_in_generation_json": "training_fingerprint" in payload,
        }
    ledger = {
        "purpose": (
            "Audit hashes of completed synthetic outputs. "
            "Do not read recomputed_training_fingerprint as a value the original run stored."
        ),
        "computed_at": utc_now(),
        "git_commit": None,
        "git_note": "The project directory was not a Git repository when this ledger was written.",
        "source_tree_fingerprint_at_audit": source_tree_fingerprint(),
        "entries": entries,
    }
    destination = ledger_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, ledger)
    print(f"Wrote {len(entries)} ledger entries to {destination}")


def _training_frame(train, meta: Path, seed: int, n_training: int):
    parts = meta.parts
    if "main" in parts:
        frame = modeling_columns(train)
    else:
        budget_name = next(part for part in parts if part.startswith("budget_"))
        budget = float(budget_name.split("_", 1)[1])
        subset = stratified_sample(train, budget_count(len(train), budget), seed)
        frame = modeling_columns(subset)
    if len(frame) != n_training:
        raise RuntimeError(f"Recomputed training rows {len(frame)} do not match {meta}: {n_training}.")
    return prepare_generator_frame(frame)


if __name__ == "__main__":
    main()

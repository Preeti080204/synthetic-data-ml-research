"""The release interface reads saved results and does not launch the held-out scorer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_project.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_default_summary_comes_from_saved_tables():
    completed = _run()
    assert completed.returncode == 0, completed.stderr
    assert "Validation runs: 390" in completed.stdout
    assert "Held-out test runs: 390" in completed.stdout
    assert "lower 30" in completed.stdout
    assert "higher 0" in completed.stdout


def test_reproduce_does_not_execute_unless_asked():
    completed = _run("reproduce")
    assert completed.returncode == 0
    assert "Not executed." in completed.stdout
    assert "run_final_test.py" in completed.stdout


def test_reproduce_refuses_a_held_out_rescore(monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_project

    def refuse(*_args, **_kwargs):
        raise AssertionError("A fit or scorer was launched.")

    monkeypatch.setattr(run_project.subprocess, "run", refuse)
    assert run_project.reproduce(["--execute", "--held-out-test"]) == 2

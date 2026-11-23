"""The saved held-out file must match the frozen plan. This test does not refit."""

import json

from experiments.cache import file_sha256
from experiments.final_guard import freeze_path
from reporting.final_audit import audit_saved_test
from utils.paths import project_root


def test_saved_test_evaluation_passes_the_integrity_audit():
    record = audit_saved_test()
    assert record["runs"] == 390
    assert record["failures"] == []
    assert record["test_based_selection"] is False


def test_audit_uses_the_published_status_when_the_authorization_file_is_absent(monkeypatch, tmp_path):
    from reporting import final_audit

    monkeypatch.setattr(final_audit, "authorization_path", lambda: tmp_path / "missing")
    record = final_audit.audit_saved_test()
    assert record["failures"] == []
    assert record["authorization_reviewer"] == "User request, 2026-11-17"
    assert record["freeze_sha256"] == "e06887586eb1e02a416e3db63bb941f9ec38b30e14dad0db444847fd177d8180"


def test_original_freeze_is_unchanged_and_post_status_is_separate():
    freeze = json.loads(freeze_path().read_text(encoding="utf-8"))
    assert freeze["test_metrics_computed"] is False
    assert freeze["final_test_authorized"] is False
    assert file_sha256(freeze_path()) == "e06887586eb1e02a416e3db63bb941f9ec38b30e14dad0db444847fd177d8180"
    status = json.loads((project_root() / "results" / "frozen" / "post_evaluation_status.json").read_text())
    assert status["held_out_evaluation"]["status"] == "completed_and_verified"
    assert status["held_out_evaluation"]["runs"] == 390
    assert status["validation_freeze_test_metrics_computed"] is False

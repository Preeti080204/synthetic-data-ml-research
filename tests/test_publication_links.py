"""Publication checks that never train models or access private documents."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _local_markdown_images(document: Path) -> list[Path]:
    content = document.read_text(encoding="utf-8")
    targets = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", content)
    paths: list[Path] = []
    for target in targets:
        target = target.split("#", 1)[0]
        if "://" in target or target.startswith("data:"):
            continue
        paths.append((document.parent / target).resolve())
    return paths


def test_readme_figure_links_exist():
    paths = _local_markdown_images(ROOT / "README.md")
    assert paths, "README should show actual research figures"
    assert all(path.is_file() for path in paths), [str(p) for p in paths if not p.is_file()]


def test_final_markdown_figure_links_exist():
    report = ROOT / "reports" / "final" / "Preeti_Gorial_Synthetic_Data_Research_Report.md"
    paths = _local_markdown_images(report)
    assert len(paths) >= 10, "Final report should include figures, not only text"
    assert all(path.is_file() for path in paths), [str(p) for p in paths if not p.is_file()]


def test_final_report_and_core_research_evidence_exist():
    assert (ROOT / "reports/final/Preeti_Gorial_Synthetic_Data_Research_Report.pdf").is_file()
    assert (ROOT / "reports/final/Preeti_Gorial_Synthetic_Data_Research_Report.docx").is_file()
    assert (ROOT / "results/final_evaluation/test_run_metrics.csv").is_file()
    assert (ROOT / "results/frozen/validation_freeze.json").is_file()
    assert (ROOT / "results/frozen/post_evaluation_status.json").is_file()

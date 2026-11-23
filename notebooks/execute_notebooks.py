"""Execute the Phase 1 notebooks with the project virtual environment."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "notebooks" / "01_data_exploration.ipynb",
    ROOT / "notebooks" / "02_baseline_analysis.ipynb",
    ROOT / "notebooks" / "03_synthetic_generation.ipynb",
    ROOT / "notebooks" / "04_main_experiments.ipynb",
    ROOT / "notebooks" / "05_quality_analysis.ipynb",
    ROOT / "notebooks" / "06_research_findings.ipynb",
]


def main() -> None:
    for path in NOTEBOOKS:
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            kernel_name="synthetic-data-ml",
            resources={"metadata": {"path": str(ROOT)}},
            timeout=600,
        )
        client.execute()
        nbformat.write(notebook, path)
        print(f"executed {path.name}")


if __name__ == "__main__":
    main()

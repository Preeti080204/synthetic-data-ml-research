"""Score the held-out test set only after every authorization check passes.

Without ``--final-test`` the process exits before the test frame is imported.
The authorization file is not created here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Final held-out test evaluation.")
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="Required. Without this flag the test set is not loaded or scored.",
    )
    args = parser.parse_args()
    if not args.final_test:
        print(
            "Refusing to compute test metrics. Re-run with --final-test only after "
            "the frozen validation plan has been reviewed and authorized.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    from experiments.final_guard import preflight

    errors = preflight()
    if errors:
        print("Refusing to score the test set.", file=sys.stderr)
        for item in errors:
            print(item, file=sys.stderr)
        raise SystemExit(2)

    from experiments.final_eval import run_authorized_test

    destination = run_authorized_test()
    print(f"Wrote test metrics to {destination}")


if __name__ == "__main__":
    main()

"""Run the CU-16 resize policy comparison experiment."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from eval.resize_policy import run_resize_policy_eval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CU-16 resize policy eval.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/golden_resize/manifest.csv"),
        help="Manifest CSV path.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-nano",
        help="Canonical VLM model for the experiment.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="How many repeats to run per image-condition pair.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ko",
        help="Prompt output language.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/eval_results"),
        help="Directory for result JSON/CSV files.",
    )
    args = parser.parse_args()

    report = run_resize_policy_eval(
        args.manifest_path,
        model=args.model,
        repeats=args.repeats,
        language=args.language,
        output_dir=args.output_dir,
    )
    LOGGER.info("Resize eval JSON: %s", report["json_path"])
    LOGGER.info("Resize eval CSV: %s", report["csv_path"])
    print(json.dumps(report["aggregates"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

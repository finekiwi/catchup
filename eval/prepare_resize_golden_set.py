"""Prepare `data/golden_resize/` and auto-generate its manifest CSV."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from eval.resize_policy import (
    collect_pdf_figures_to_dataset,
    discover_manifest_rows,
    write_manifest_csv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CU-16 resize golden set.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/golden"),
        help="Directory containing source golden PDFs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/golden_resize"),
        help="Bucketed dataset root directory.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/golden_resize/manifest.csv"),
        help="Manifest CSV output path.",
    )
    parser.add_argument(
        "--classify-model",
        type=str,
        default="gpt-4.1-nano",
        help="Model used to auto-bucket PDF-derived figures.",
    )
    args = parser.parse_args()

    copied = collect_pdf_figures_to_dataset(
        args.source_dir,
        args.output_dir,
        classify_model=args.classify_model,
    )
    rows = discover_manifest_rows(args.output_dir)
    manifest_path = write_manifest_csv(rows, args.manifest_path)

    LOGGER.info("Copied %d PDF-derived figures into %s", len(copied), args.output_dir)
    LOGGER.info("Manifest saved: %s (%d rows)", manifest_path, len(rows))


if __name__ == "__main__":
    main()

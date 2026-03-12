"""Index golden set documents into both baseline and CatchUp collections.

Run this once before executing before_after.py:
    python -m eval.setup_golden_index

Documents are read from data/golden/. Both collections are populated:
  - catchup_baseline: PyPDF flat extraction (via eval.baseline.index_baseline)
  - catchup_rag:      CatchUp structured parsing (via parsers.pdf_parser + rag.qa_chain)
"""

from __future__ import annotations

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)

GOLDEN_DIR = Path("data/golden")


def index_all_baseline() -> None:
    """Index all PDFs in data/golden/ into the baseline collection."""
    from eval.baseline import index_baseline

    pdfs = sorted(GOLDEN_DIR.glob("*.pdf"))
    LOGGER.info("Indexing %d PDFs into baseline collection...", len(pdfs))
    for pdf in pdfs:
        LOGGER.info("  baseline: %s", pdf.name)
        index_baseline(pdf)
    LOGGER.info("Baseline indexing complete.")


def index_all_catchup() -> None:
    """Parse and index all PDFs in data/golden/ into the CatchUp collection."""
    from parsers.pdf_parser import parse_pdf
    from rag.qa_chain import index_document

    pdfs = sorted(GOLDEN_DIR.glob("*.pdf"))
    LOGGER.info("Indexing %d PDFs into CatchUp collection...", len(pdfs))
    for pdf in pdfs:
        LOGGER.info("  catchup: %s", pdf.name)
        try:
            doc = parse_pdf(str(pdf))
            index_document(doc)
            LOGGER.info("    -> %d blocks indexed", len(doc.blocks))
        except Exception:
            LOGGER.exception("Failed to index %s via CatchUp pipeline", pdf.name)
    LOGGER.info("CatchUp indexing complete.")


def main() -> None:
    if not GOLDEN_DIR.exists():
        raise FileNotFoundError(f"Golden documents directory not found: {GOLDEN_DIR}")

    pdfs = list(GOLDEN_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {GOLDEN_DIR}")

    LOGGER.info("Found %d PDF(s) in %s: %s", len(pdfs), GOLDEN_DIR, [p.name for p in pdfs])

    index_all_baseline()
    index_all_catchup()

    LOGGER.info("All documents indexed. Run: python -m eval.before_after")


if __name__ == "__main__":
    main()

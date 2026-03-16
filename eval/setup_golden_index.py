"""Index golden set documents into baseline and CatchUp collections.

Run this once before executing before_after.py:
    python -m eval.setup_golden_index

Documents are read from data/golden/. Three collections are populated per format:
  PDF:
    - catchup_baseline:      PyPDF flat extraction, 1000-char chunks
    - catchup_rag:           CatchUp structured parsing, Docling block-level (variable size)
    - catchup_rag_chunked:   CatchUp structured parsing, rechunked to 1000-char (fair comparison)
  ipynb:
    - catchup_baseline:      nbformat flat extraction, 1000-char chunks (same collection, different source)
    - catchup_rag:           CatchUp ipynb_parser structured parsing, block-level
    - catchup_rag_chunked:   CatchUp ipynb_parser + rechunked to HybridChunker max_tokens=500
"""

from __future__ import annotations

import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
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


def index_all_catchup_chunked() -> None:
    """Parse and index all PDFs using CatchUp parsing + rechunked (1000-char) chunks."""
    from parsers.pdf_parser import parse_pdf
    from rag.qa_chain import index_document_chunked

    pdfs = sorted(GOLDEN_DIR.glob("*.pdf"))
    LOGGER.info("Indexing %d PDFs into CatchUp-chunked collection...", len(pdfs))
    for pdf in pdfs:
        LOGGER.info("  catchup-chunked: %s", pdf.name)
        try:
            doc = parse_pdf(str(pdf))
            index_document_chunked(doc)
            LOGGER.info(
                "    -> %d blocks parsed, rechunked and indexed", len(doc.blocks)
            )
        except Exception:
            LOGGER.exception(
                "Failed to index %s via CatchUp-chunked pipeline", pdf.name
            )
    LOGGER.info("CatchUp-chunked indexing complete.")


def index_all_baseline_ipynb() -> None:
    """Index all .ipynb files in data/golden/ into the baseline collection (nbformat flat extraction)."""
    from eval.baseline import index_baseline_ipynb

    notebooks = sorted(GOLDEN_DIR.glob("*.ipynb"))
    LOGGER.info("Indexing %d notebooks into baseline collection...", len(notebooks))
    for nb in notebooks:
        LOGGER.info("  baseline-ipynb: %s", nb.name)
        index_baseline_ipynb(nb)
    LOGGER.info("Baseline ipynb indexing complete.")


def index_all_catchup_ipynb() -> None:
    """Parse and index all .ipynb files using CatchUp ipynb_parser into the CatchUp collection."""
    from parsers.ipynb_parser import parse_ipynb
    from rag.qa_chain import index_document

    notebooks = sorted(GOLDEN_DIR.glob("*.ipynb"))
    LOGGER.info("Indexing %d notebooks into CatchUp collection...", len(notebooks))
    for nb in notebooks:
        LOGGER.info("  catchup-ipynb: %s", nb.name)
        try:
            doc = parse_ipynb(str(nb))
            index_document(doc)
            LOGGER.info("    -> %d blocks indexed", len(doc.blocks))
        except Exception:
            LOGGER.exception("Failed to index %s via CatchUp ipynb pipeline", nb.name)
    LOGGER.info("CatchUp ipynb indexing complete.")


def index_all_catchup_chunked_ipynb() -> None:
    """Parse and index all .ipynb files using CatchUp ipynb_parser + HybridChunker rechunking."""
    from parsers.ipynb_parser import parse_ipynb
    from rag.qa_chain import index_document_chunked

    notebooks = sorted(GOLDEN_DIR.glob("*.ipynb"))
    LOGGER.info(
        "Indexing %d notebooks into CatchUp-chunked collection...", len(notebooks)
    )
    for nb in notebooks:
        LOGGER.info("  catchup-chunked-ipynb: %s", nb.name)
        try:
            doc = parse_ipynb(str(nb))
            index_document_chunked(doc)
            LOGGER.info(
                "    -> %d blocks parsed, rechunked and indexed", len(doc.blocks)
            )
        except Exception:
            LOGGER.exception(
                "Failed to index %s via CatchUp-chunked ipynb pipeline", nb.name
            )
    LOGGER.info("CatchUp-chunked ipynb indexing complete.")


def main() -> None:
    if not GOLDEN_DIR.exists():
        raise FileNotFoundError(f"Golden documents directory not found: {GOLDEN_DIR}")

    pdfs = list(GOLDEN_DIR.glob("*.pdf"))
    notebooks = list(GOLDEN_DIR.glob("*.ipynb"))

    if not pdfs and not notebooks:
        raise FileNotFoundError(f"No PDF or .ipynb files found in {GOLDEN_DIR}")

    if pdfs:
        LOGGER.info("Found %d PDF(s): %s", len(pdfs), [p.name for p in pdfs])
        index_all_baseline()
        index_all_catchup()
        index_all_catchup_chunked()

    if notebooks:
        LOGGER.info(
            "Found %d notebook(s): %s", len(notebooks), [n.name for n in notebooks]
        )
        index_all_baseline_ipynb()
        index_all_catchup_ipynb()
        index_all_catchup_chunked_ipynb()

    LOGGER.info("All documents indexed. Run: python -m eval.before_after")


if __name__ == "__main__":
    main()

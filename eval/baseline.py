"""Baseline RAG pipeline using PyPDF simple text extraction + fixed-size chunking.

This is the 'Before' pipeline for Before/After comparison with CatchUp.
Controls:
  - Same embedding: text-embedding-3-small
  - Same top-k: 5
  - Same LLM
  - temperature=0.0, seed fixed
  - Only variable: parsing quality (PyPDF flat text vs CatchUp structured blocks)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from db.chroma import _build_client
from prompts.rag_qa import PROMPT
from rag.qa_chain import (
    EMBED_MODEL,
    QAResult,
    SourceBlock,
    _call_openai,
    _get_openai_embedding,
)
from utils.logging import log_api_call

load_dotenv()

LOGGER = logging.getLogger(__name__)

BASELINE_COLLECTION = "catchup_baseline"
_EMBED_COST_PER_1M_USD = 0.02  # USD per 1M tokens for text-embedding-3-small

# Baseline only supports OpenAI models — non-OpenAI providers use _call_openai internally
_SUPPORTED_BASELINE_MODELS: frozenset[str] = frozenset({
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-5-nano",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BaselineChunk:
    """A fixed-size text chunk produced from flat PyPDF extraction.

    Attributes:
        chunk_id: Unique identifier in the format ``{source}:{chunk_index}``.
        source: Original PDF filename (basename only).
        content: Raw text content of the chunk.
        chunk_index: Zero-based sequential index within the document.
    """

    chunk_id: str
    source: str
    content: str
    chunk_index: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_baseline_collection() -> Optional[Any]:
    """Get or create the baseline ChromaDB collection."""
    try:
        client = _build_client()
        return client.get_or_create_collection(name=BASELINE_COLLECTION)
    except Exception:
        LOGGER.exception("Failed to initialize baseline ChromaDB collection")
        return None


# ---------------------------------------------------------------------------
# Core pipeline functions
# ---------------------------------------------------------------------------


def extract_text_pypdf(pdf_path: Path) -> str:
    """Extract all text from a PDF file using PyPDF flat extraction.

    This intentionally uses the simplest possible extraction strategy — no layout
    analysis, no table detection, no image parsing — to represent the 'Before'
    baseline. All pages are concatenated with newlines.

    Args:
        pdf_path: Path to the PDF file to extract.

    Returns:
        Concatenated plain text from all pages.

    Raises:
        ImportError: If pypdf is not installed.
        FileNotFoundError: If pdf_path does not exist.
    """
    try:
        import pypdf as _pypdf
    except ImportError as exc:
        raise ImportError("pypdf not installed. Run: uv add pypdf") from exc

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = _pypdf.PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return "\n".join(pages)


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 1000,
    overlap: int = 100,
) -> list[BaselineChunk]:
    """Split flat text into fixed-size overlapping chunks.

    Uses character-level sliding window with the given overlap. Empty chunks
    (after stripping) are skipped.

    Args:
        text: Full document text to split.
        source: Original filename to embed in each BaselineChunk.
        chunk_size: Maximum characters per chunk.
        overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List of BaselineChunk instances in document order.
    """
    if not text.strip():
        return []

    chunks: list[BaselineChunk] = []
    step = max(1, chunk_size - overlap)
    chunk_index = 0

    for start in range(0, len(text), step):
        end = start + chunk_size
        segment = text[start:end].strip()
        if not segment:
            continue

        chunks.append(BaselineChunk(
            chunk_id=f"{source}:{chunk_index}",
            source=source,
            content=segment,
            chunk_index=chunk_index,
        ))
        chunk_index += 1

        # Stop if we've consumed the end of the text
        if end >= len(text):
            break

    return chunks


def index_baseline(pdf_path: Path) -> None:
    """Extract, chunk, embed, and store a PDF into the baseline ChromaDB collection.

    Uses PyPDF flat extraction → fixed-size chunking → text-embedding-3-small.
    If the source has already been indexed (any chunk present), the call is a no-op.

    Args:
        pdf_path: Path to the PDF to index.
    """
    collection = _get_baseline_collection()
    if collection is None:
        LOGGER.error("Baseline collection unavailable — skipping index for %s", pdf_path)
        return

    source = pdf_path.name

    # Check if already indexed: any chunk with this source present
    try:
        existing = collection.get(where={"source": source})
        if existing.get("ids"):
            LOGGER.info("Baseline: %s already indexed (%d chunks), skipping", source, len(existing["ids"]))
            return
    except Exception:
        LOGGER.debug("Could not check baseline index for %s — proceeding", source)

    LOGGER.info("Baseline: indexing %s", pdf_path)

    try:
        raw_text = extract_text_pypdf(pdf_path)
    except Exception as exc:
        LOGGER.error("PyPDF extraction failed for %s: %s", pdf_path, exc)
        return

    chunks = chunk_text(raw_text, source)
    if not chunks:
        LOGGER.warning("Baseline: no chunks produced for %s", pdf_path)
        return

    LOGGER.info("Baseline: %d chunks from %s", len(chunks), source)

    for chunk in chunks:
        t0 = time.perf_counter()
        try:
            vector, total_tokens = _get_openai_embedding(chunk.content)
            latency_ms = (time.perf_counter() - t0) * 1000
            log_api_call(
                model=EMBED_MODEL,
                stage="baseline_embed",
                input_tokens=total_tokens,
                output_tokens=0,
                latency_ms=latency_ms,
                cost_usd=total_tokens * _EMBED_COST_PER_1M_USD / 1_000_000,
                success=True,
            )
        except Exception as exc:
            LOGGER.warning("Embedding failed for chunk %s: %s", chunk.chunk_id, exc)
            log_api_call(
                model=EMBED_MODEL,
                stage="baseline_embed",
                input_tokens=0,
                output_tokens=0,
                latency_ms=(time.perf_counter() - t0) * 1000,
                cost_usd=0.0,
                success=False,
                error=str(exc),
            )
            continue

        metadata: dict[str, Any] = {
            "source": chunk.source,
            "chunk_index": chunk.chunk_index,
        }

        try:
            collection.upsert(
                ids=[chunk.chunk_id],
                documents=[chunk.content],
                metadatas=[metadata],
                embeddings=[vector],
            )
        except Exception:
            LOGGER.exception("Failed to upsert baseline chunk %s", chunk.chunk_id)


def query_baseline(
    question: str,
    top_k: int = 5,
    model: str = "gpt-4o-mini",
) -> QAResult:
    """Answer a question using the baseline RAG pipeline.

    Embeds the question with text-embedding-3-small, retrieves top-k chunks from
    the baseline collection, and generates an answer with the same LLM and prompt
    as the CatchUp pipeline. Only OpenAI models are supported for the baseline.

    Args:
        question: Natural language question.
        top_k: Number of baseline chunks to retrieve.
        model: OpenAI model identifier for answer generation.

    Returns:
        QAResult identical in schema to the CatchUp pipeline output.

    Raises:
        ValueError: If model is not an OpenAI model supported by the baseline.
    """
    if model not in _SUPPORTED_BASELINE_MODELS:
        raise ValueError(
            f"Baseline pipeline only supports OpenAI models. Got: {model!r}. "
            f"Supported: {sorted(_SUPPORTED_BASELINE_MODELS)}"
        )

    t_total = time.perf_counter()

    collection = _get_baseline_collection()
    if collection is None or collection.count() == 0:
        return QAResult(
            question=question,
            answer="No baseline documents indexed. Run index_baseline() first.",
            source_blocks=[],
            model=model,
            latency_ms=(time.perf_counter() - t_total) * 1000,
            input_tokens=0,
            output_tokens=0,
        )

    # Embed question
    try:
        t_embed = time.perf_counter()
        question_vector, embed_tokens = _get_openai_embedding(question)
        embed_latency_ms = (time.perf_counter() - t_embed) * 1000
        log_api_call(
            model=EMBED_MODEL,
            stage="baseline_embed",
            input_tokens=embed_tokens,
            output_tokens=0,
            latency_ms=embed_latency_ms,
            cost_usd=embed_tokens * _EMBED_COST_PER_1M_USD / 1_000_000,
            success=True,
        )
    except Exception as exc:
        LOGGER.error("Failed to embed question for baseline: %s", exc)
        log_api_call(
            model=EMBED_MODEL,
            stage="baseline_embed",
            input_tokens=0,
            output_tokens=0,
            latency_ms=(time.perf_counter() - t_total) * 1000,
            cost_usd=0.0,
            success=False,
            error=str(exc),
        )
        return QAResult(
            question=question,
            answer="Failed to process question.",
            source_blocks=[],
            model=model,
            latency_ms=(time.perf_counter() - t_total) * 1000,
            input_tokens=0,
            output_tokens=0,
        )

    # Search baseline collection
    try:
        n_results = min(top_k, collection.count())
        raw_results = collection.query(
            query_embeddings=[question_vector],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        LOGGER.exception("Baseline ChromaDB search failed")
        raw_results = {}

    ids = (raw_results.get("ids") or [[]])[0]
    contents = (raw_results.get("documents") or [[]])[0]
    metadatas = (raw_results.get("metadatas") or [[]])[0]

    if not ids:
        return QAResult(
            question=question,
            answer="No relevant baseline chunks found.",
            source_blocks=[],
            model=model,
            latency_ms=(time.perf_counter() - t_total) * 1000,
            input_tokens=0,
            output_tokens=0,
        )

    # Build source blocks and context string
    source_blocks: list[SourceBlock] = []
    context_parts: list[str] = []

    for i, content in enumerate(contents):
        meta = metadatas[i] if i < len(metadatas) else {}
        source = meta.get("source", "unknown")
        chunk_index = meta.get("chunk_index", 0)

        source_blocks.append(SourceBlock(
            document_id="",
            source=source,
            block_order=chunk_index,
            block_type="text",
            content_preview=content[:200],
            page=None,
            cell_index=None,
        ))

        context_parts.append(f"[{source}] chunk {chunk_index}\n{content}")

    context = "\n\n---\n\n".join(context_parts)
    user_content = f"Context:\n{context}\n\nQuestion: {question}"

    # LLM answer generation — baseline only supports OpenAI for controlled comparison
    try:
        t_llm = time.perf_counter()
        raw_answer, input_tokens, output_tokens = _call_openai(model, PROMPT, user_content)
        llm_latency_ms = (time.perf_counter() - t_llm) * 1000

        log_api_call(
            model=model,
            stage="baseline_generate",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=llm_latency_ms,
            cost_usd=0.0,  # cost calculation omitted for baseline simplicity
            success=True,
        )
        return QAResult(
            question=question,
            answer=raw_answer.strip(),
            source_blocks=source_blocks,
            model=model,
            latency_ms=(time.perf_counter() - t_total) * 1000,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    except Exception as exc:
        LOGGER.error("Baseline LLM generation failed: %s", exc)
        log_api_call(
            model=model,
            stage="baseline_generate",
            input_tokens=0,
            output_tokens=0,
            latency_ms=(time.perf_counter() - t_total) * 1000,
            cost_usd=0.0,
            success=False,
            error=str(exc),
        )
        return QAResult(
            question=question,
            answer="Baseline answer generation failed.",
            source_blocks=source_blocks,
            model=model,
            latency_ms=(time.perf_counter() - t_total) * 1000,
            input_tokens=0,
            output_tokens=0,
        )


def extract_text_ipynb(ipynb_path: Path) -> str:
    """Extract plain text from a Jupyter notebook via nbformat.

    Concatenates markdown cell source, code cell source, and text outputs
    in notebook order. This intentionally ignores rich outputs (images, HTML)
    to represent the flat-text 'Before' baseline for ipynb.

    Args:
        ipynb_path: Path to the .ipynb file.

    Returns:
        Concatenated plain text from all cells and their text outputs.

    Raises:
        ImportError: If nbformat is not installed.
        FileNotFoundError: If ipynb_path does not exist.
    """
    try:
        import nbformat as _nbformat
    except ImportError as exc:
        raise ImportError("nbformat not installed. Run: uv add nbformat") from exc

    if not ipynb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {ipynb_path}")

    with open(ipynb_path, encoding="utf-8") as f:
        nb = _nbformat.read(f, as_version=4)

    parts: list[str] = []
    for cell in nb.cells:
        src = "".join(cell.get("source", []) if isinstance(cell.get("source"), list) else [cell.get("source", "")])
        if src.strip():
            parts.append(src)

        # Include text outputs from code cells
        if cell.cell_type == "code":
            for output in cell.get("outputs", []):
                otype = output.get("output_type", "")
                text_content = ""
                if otype in ("stream", "error"):
                    text_content = "".join(output.get("text", []))
                elif otype in ("execute_result", "display_data"):
                    data = output.get("data", {})
                    text_content = "".join(data.get("text/plain", []))
                if text_content.strip():
                    parts.append(text_content)

    return "\n\n".join(parts)


def index_baseline_ipynb(ipynb_path: Path) -> None:
    """Extract, chunk, embed, and store a notebook into the baseline ChromaDB collection.

    Uses nbformat flat extraction → fixed-size chunking → text-embedding-3-small.
    If the source has already been indexed, the call is a no-op.

    Args:
        ipynb_path: Path to the .ipynb file to index.
    """
    collection = _get_baseline_collection()
    if collection is None:
        LOGGER.error("Baseline collection unavailable — skipping index for %s", ipynb_path)
        return

    source = ipynb_path.name

    try:
        existing = collection.get(where={"source": source})
        if existing.get("ids"):
            LOGGER.info("Baseline: %s already indexed (%d chunks), skipping", source, len(existing["ids"]))
            return
    except Exception:
        LOGGER.debug("Could not check baseline index for %s — proceeding", source)

    LOGGER.info("Baseline: indexing notebook %s", ipynb_path)

    try:
        raw_text = extract_text_ipynb(ipynb_path)
    except Exception as exc:
        LOGGER.error("nbformat extraction failed for %s: %s", ipynb_path, exc)
        return

    chunks = chunk_text(raw_text, source)
    if not chunks:
        LOGGER.warning("Baseline: no chunks produced for %s", source)
        return

    LOGGER.info("Baseline: %d chunks from %s", len(chunks), source)

    for chunk in chunks:
        t0 = time.perf_counter()
        try:
            vector, total_tokens = _get_openai_embedding(chunk.content)
            latency_ms = (time.perf_counter() - t0) * 1000
            log_api_call(
                model=EMBED_MODEL,
                stage="baseline_embed",
                input_tokens=total_tokens,
                output_tokens=0,
                latency_ms=latency_ms,
                cost_usd=total_tokens * _EMBED_COST_PER_1M_USD / 1_000_000,
                success=True,
            )
        except Exception as exc:
            LOGGER.warning("Embedding failed for chunk %s: %s", chunk.chunk_id, exc)
            log_api_call(
                model=EMBED_MODEL,
                stage="baseline_embed",
                input_tokens=0,
                output_tokens=0,
                latency_ms=(time.perf_counter() - t0) * 1000,
                cost_usd=0.0,
                success=False,
                error=str(exc),
            )
            continue

        metadata: dict[str, Any] = {
            "source": chunk.source,
            "chunk_index": chunk.chunk_index,
        }

        try:
            collection.upsert(
                ids=[chunk.chunk_id],
                documents=[chunk.content],
                metadatas=[metadata],
                embeddings=[vector],
            )
        except Exception:
            LOGGER.exception("Failed to upsert baseline chunk %s", chunk.chunk_id)


__all__ = [
    "BASELINE_COLLECTION",
    "BaselineChunk",
    "extract_text_pypdf",
    "extract_text_ipynb",
    "chunk_text",
    "index_baseline",
    "index_baseline_ipynb",
    "query_baseline",
]

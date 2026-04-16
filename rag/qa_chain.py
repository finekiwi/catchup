"""RAG Q&A pipeline: index Document blocks into ChromaDB and answer questions with LLM.

Workflow:
  1. index_document() embeds each block with text-embedding-3-small and stores in ChromaDB.
  2. query() embeds the question, retrieves top-k blocks, and generates an answer with the LLM.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from db.chroma import _build_client
from llm.note_generator import _is_noise_block
from models.document import Document
from prompts.rag_qa import PROMPT
from rag.query_rewriter import rewrite_query as _rewrite_query
from utils.embed import get_openai_embedding as _get_openai_embedding
from utils.logging import log_api_call
from utils.models import MODEL_REGISTRY, call_llm, compute_cost

load_dotenv()

LOGGER = logging.getLogger(__name__)

RAG_COLLECTION_NAME = "catchup_rag"
RAG_CHUNKED_COLLECTION_NAME = "catchup_rag_chunked"
EMBED_MODEL = "text-embedding-3-small"
_EMBED_COST_PER_1M_USD = 0.02  # USD per 1M tokens for text-embedding-3-small

# Bump this string whenever noise filter logic, chunking strategy, or embedding model changes.
# index_document() / index_document_chunked() will auto-delete and re-index documents
# whose stored version does not match.
INDEXING_VERSION = "v2"

# HybridChunker max_tokens for controlled comparison with baseline
# cl100k_base: ~4 chars/token → 500 tokens ≈ 2000 chars
_RECHUNK_MAX_TOKENS = 500

SUPPORTED_MODELS: list[str] = list(MODEL_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------


class SourceBlock(BaseModel):
    """A retrieved document block used as evidence for an answer."""

    document_id: str
    source: str  # original filename
    block_order: int
    block_type: str
    content_preview: str  # first 200 chars
    page: Optional[int] = None
    cell_index: Optional[int] = None
    image_path: Optional[str] = None


class QAResult(BaseModel):
    """Result of a RAG Q&A query."""

    question: str
    answer: str
    source_blocks: list[SourceBlock] = Field(default_factory=list)
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    rewritten_query: str | None = None


def _get_rag_collection(name: str = RAG_COLLECTION_NAME) -> Optional[Any]:
    """Get or create a RAG ChromaDB collection by name."""
    try:
        client = _build_client()
        return client.get_or_create_collection(name=name)
    except Exception:
        LOGGER.exception("Failed to initialize RAG ChromaDB collection: %s", name)
        return None


def rechunk_blocks(
    document: "Document",
    max_tokens: int = _RECHUNK_MAX_TOKENS,
) -> list[tuple[str, dict]]:
    """Chunk a Document using Docling HybridChunker for fair comparison with baseline.

    Loads the cached DoclingDocument (saved by pdf_parser.py at parse time) and runs
    HybridChunker with an OpenAI tiktoken tokenizer. Falls back to flat block iteration
    if the DoclingDocument cache is unavailable.

    Args:
        document: Parsed CatchUp Document (used for source/id metadata and fallback).
        max_tokens: Maximum tokens per chunk (default 500 ≈ 2000 chars for cl100k_base).

    Returns:
        List of (chunk_text, metadata_dict) tuples in document order.
    """
    from pathlib import Path as _Path
    from utils.cache import load_docling_doc

    # Attempt to load cached DoclingDocument for HybridChunker
    dl_doc = None
    # Reconstruct the original file path from source name via known golden dir
    # (cache lookup is hash-based; we search data/golden/ for a matching filename)
    for candidate_dir in ("data/golden", "data"):
        candidate = _Path(candidate_dir) / document.source
        if candidate.exists():
            dl_doc = load_docling_doc(candidate)
            if dl_doc is not None:
                break

    if dl_doc is not None:
        try:
            import tiktoken
            from docling_core.transforms.chunker import HybridChunker
            from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

            enc = tiktoken.get_encoding("cl100k_base")
            tok = OpenAITokenizer(tokenizer=enc, max_tokens=max_tokens)
            chunker = HybridChunker(tokenizer=tok)

            chunks: list[tuple[str, dict]] = []
            for i, chunk in enumerate(chunker.chunk(dl_doc)):
                text = chunk.text.strip()
                if not text:
                    continue
                # Extract page from first doc_item provenance if available
                page: Optional[int] = None
                doc_items = getattr(chunk.meta, "doc_items", []) or []
                if doc_items:
                    prov = getattr(doc_items[0], "prov", None)
                    if prov:
                        first = prov[0] if isinstance(prov, list) else prov
                        try:
                            page = int(getattr(first, "page_no", None) or 0) or None
                        except (TypeError, ValueError):
                            pass
                meta: dict[str, Any] = {
                    "document_id": document.id,
                    "source": document.source,
                    "block_order": i,
                    "block_type": "text",
                    "chunk_index": i,
                }
                if page is not None:
                    meta["page"] = page
                chunks.append((text, meta))

            LOGGER.info(
                "HybridChunker: %d chunks from %s (max_tokens=%d)",
                len(chunks),
                document.source,
                max_tokens,
            )
            return chunks

        except Exception as exc:
            LOGGER.warning(
                "HybridChunker failed for %s, falling back: %s", document.source, exc
            )

    # Fallback: flat block iteration with chunk cache for ipynb files
    LOGGER.warning(
        "rechunk_blocks: DoclingDocument not cached for %s — using flat blocks",
        document.source,
    )
    from utils.cache import load_cached_chunks, save_cached_chunks

    # Try loading previously computed flat chunks to avoid re-iteration on re-runs
    source_path: Optional["_Path"] = None
    for candidate_dir in ("data/golden", "data"):
        candidate = _Path(candidate_dir) / document.source
        if candidate.exists():
            source_path = candidate
            break

    if source_path is not None:
        cached_chunks = load_cached_chunks(source_path)
        if cached_chunks is not None:
            return cached_chunks

    flat: list[tuple[str, dict]] = []
    for block in document.blocks:
        content = block.content.strip()
        if not content or _is_noise_block(block):
            continue
        meta = {
            "document_id": document.id,
            "source": document.source,
            "block_order": block.order,
            "block_type": block.type.value,
        }
        if block.metadata.page is not None:
            meta["page"] = block.metadata.page
        flat.append((content, meta))

    if source_path is not None:
        save_cached_chunks(source_path, flat)

    return flat



def _expand_with_adjacent_blocks(
    collection: Any,
    hit_metadatas: list[dict],
    hit_contents: list[str],
    context_window: int = 2,
) -> list[tuple[dict, str]]:
    """Expand top-k hits by fetching adjacent blocks (±context_window) from the same document.

    Each hit block is augmented with its neighbours so the LLM receives a wider
    contiguous passage. Blocks from different documents remain separated.
    Deduplication is done by (document_id, block_order) key.

    Args:
        collection: ChromaDB collection to fetch from.
        hit_metadatas: Metadata dicts from the top-k query results.
        hit_contents: Text content corresponding to hit_metadatas.
        context_window: Number of blocks to expand in each direction (default 2).

    Returns:
        List of (metadata, content) tuples sorted by (document_id, block_order),
        including both original hits and their neighbours.
    """
    # Build fetch targets: doc_id → set of block_orders to retrieve
    fetch_targets: dict[str, set[int]] = {}
    for meta in hit_metadatas:
        doc_id = meta.get("document_id", "")
        if not doc_id:
            continue
        order = int(meta.get("block_order", 0))
        orders = fetch_targets.setdefault(doc_id, set())
        for delta in range(-context_window, context_window + 1):
            adj = order + delta
            if adj >= 0:
                orders.add(adj)

    # Seed seen set with original hits
    seen: set[str] = set()
    expanded: list[tuple[dict, str]] = []
    for meta, content in zip(hit_metadatas, hit_contents):
        key = f"{meta.get('document_id', '')}:{meta.get('block_order', 0)}"
        if key not in seen:
            seen.add(key)
            expanded.append((meta, content))

    # Fetch adjacent blocks from ChromaDB
    for doc_id, orders in fetch_targets.items():
        try:
            result = collection.get(
                where={
                    "$and": [
                        {"document_id": {"$eq": doc_id}},
                        {"block_order": {"$in": sorted(orders)}},
                    ]
                },
                include=["documents", "metadatas"],
            )
        except Exception:
            LOGGER.debug("Adjacent block fetch failed for doc_id=%s", doc_id)
            continue

        fetched_contents: list[str] = result.get("documents") or []
        fetched_metas: list[dict] = result.get("metadatas") or []
        for meta, content in zip(fetched_metas, fetched_contents):
            key = f"{meta.get('document_id', '')}:{meta.get('block_order', 0)}"
            if key not in seen:
                seen.add(key)
                expanded.append((meta, content))

    # Sort for coherent context: group by document, ordered within document
    expanded.sort(
        key=lambda x: (x[0].get("document_id", ""), int(x[0].get("block_order", 0)))
    )
    return expanded


def _get_stored_indexing_version(collection: Any, document_id: str) -> str | None:
    """Return the indexing_version stored for the first chunk of document_id, or None."""
    try:
        result = collection.get(
            where={"document_id": document_id},
            limit=1,
            include=["metadatas"],
        )
        metas = result.get("metadatas") or []
        if metas:
            return metas[0].get("indexing_version")
    except Exception:
        pass
    return None


def _is_document_indexed(
    collection: Any, document_id: str, expected_block_count: int
) -> bool:
    """Return True only if all expected blocks for this document_id are already stored.

    Comparing stored count against expected_block_count prevents partial ingests
    from being treated as complete — if any block failed on a prior run, the document
    will be re-indexed and the missing blocks backfilled via upsert.
    """
    try:
        result = collection.get(where={"document_id": document_id})
        return len(result.get("ids", [])) >= expected_block_count
    except Exception:
        return False


def _call_openai(model: str, system: str, user: str) -> tuple[str, int, int]:
    """Compatibility wrapper for modules that still import rag.qa_chain._call_openai."""
    response = call_llm(
        model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=2048,
    )
    return response.content, response.input_tokens, response.output_tokens


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def index_document(document: Document) -> None:
    """Embed and store all blocks of a Document in ChromaDB.

    Each block is stored with metadata: document_id, source, block_order, block_type,
    and optionally page / cell_index. If the document_id is already indexed, the call
    is a no-op (cache).

    Blocks that fail to embed are skipped individually — other blocks still get stored.

    Args:
        document: Source Document whose blocks will be embedded and stored.
    """
    collection = _get_rag_collection()
    if collection is None:
        LOGGER.error(
            "RAG collection unavailable — skipping index for document id=%s",
            document.id,
        )
        return

    indexable_blocks = [
        b for b in document.blocks if b.content.strip() and not _is_noise_block(b)
    ]

    # Guard: if noise filtering removed every block there is nothing to embed.
    # Treat this as a no-op rather than letting expected_block_count==0 cause
    # _is_document_indexed to always return True and silently suppress future attempts.
    if not indexable_blocks:
        LOGGER.warning(
            "index_document skipped for document id=%s: all blocks filtered as noise",
            document.id,
        )
        return

    # Auto-reindex if the stored indexing version differs from the current one.
    # stored_version is None when the document was indexed before versioning was introduced
    # (no indexing_version field in metadata) — treat as stale and re-index.
    stored_version = _get_stored_indexing_version(collection, document.id)
    if stored_version != INDEXING_VERSION:
        try:
            existing_ids = collection.get(where={"document_id": document.id}).get("ids", [])
            if existing_ids:
                LOGGER.info(
                    "Indexing version mismatch for document id=%s (stored=%r, current=%r) — re-indexing",
                    document.id,
                    stored_version,
                    INDEXING_VERSION,
                )
                collection.delete(ids=existing_ids)
        except Exception:
            LOGGER.warning("Failed to delete stale vectors for document id=%s", document.id)

    if _is_document_indexed(collection, document.id, len(indexable_blocks)):
        LOGGER.info("Document id=%s already indexed, skipping", document.id)
        return

    for block in indexable_blocks:
        content = block.content.strip()

        t0 = time.perf_counter()
        try:
            vector, total_tokens = _get_openai_embedding(content)
            latency_ms = (time.perf_counter() - t0) * 1000
            log_api_call(
                model=EMBED_MODEL,
                stage="rag_embed",
                input_tokens=total_tokens,
                output_tokens=0,
                latency_ms=latency_ms,
                cost_usd=total_tokens * _EMBED_COST_PER_1M_USD / 1_000_000,
                success=True,
            )
        except Exception as exc:
            LOGGER.warning(
                "Embedding failed for block order=%d, document id=%s: %s",
                block.order,
                document.id,
                exc,
            )
            log_api_call(
                model=EMBED_MODEL,
                stage="rag_embed",
                input_tokens=0,
                output_tokens=0,
                latency_ms=(time.perf_counter() - t0) * 1000,
                cost_usd=0.0,
                success=False,
                error=str(exc),
            )
            continue

        metadata: dict[str, Any] = {
            "document_id": document.id,
            "source": document.source,
            "block_order": block.order,
            "block_type": block.type.value,
            "indexing_version": INDEXING_VERSION,
        }
        if block.metadata.page is not None:
            metadata["page"] = block.metadata.page
        if block.metadata.cell_index is not None:
            metadata["cell_index"] = block.metadata.cell_index
        if block.image_path is not None:
            metadata["image_path"] = block.image_path

        try:
            collection.upsert(
                ids=[f"{document.id}:{block.order}"],
                documents=[content],
                metadatas=[metadata],
                embeddings=[vector],
            )
        except Exception:
            LOGGER.exception(
                "Failed to upsert block order=%d into ChromaDB, document id=%s",
                block.order,
                document.id,
            )


def index_document_chunked(document: Document) -> None:
    """Embed and store rechunked blocks of a Document in the chunked RAG collection.

    Same as index_document() but applies rechunk_blocks() first so chunk sizes
    match the baseline (1000 chars, 100 overlap). Use this for controlled comparison.

    Args:
        document: Source Document whose blocks will be rechunked, embedded, and stored.
    """
    collection = _get_rag_collection(RAG_CHUNKED_COLLECTION_NAME)
    if collection is None:
        LOGGER.error(
            "Chunked RAG collection unavailable — skipping document id=%s", document.id
        )
        return

    chunks = rechunk_blocks(document)
    if not chunks:
        LOGGER.warning("No chunks produced for document id=%s", document.id)
        return

    # Auto-reindex if stored indexing version differs (including None = pre-versioning), then
    # skip if already up-to-date.
    try:
        existing = collection.get(where={"document_id": document.id}, include=["metadatas"])
        existing_ids = existing.get("ids", [])
        stored_version = (existing.get("metadatas") or [{}])[0].get("indexing_version") if existing_ids else None
        if stored_version != INDEXING_VERSION:
            if existing_ids:
                LOGGER.info(
                    "Chunked indexing version mismatch for document id=%s (stored=%r, current=%r) — re-indexing",
                    document.id,
                    stored_version,
                    INDEXING_VERSION,
                )
                collection.delete(ids=existing_ids)
            existing_ids = []
        if len(existing_ids) >= len(chunks):
            LOGGER.info("Chunked document id=%s already indexed, skipping", document.id)
            return
    except Exception:
        pass

    for idx, (content, metadata) in enumerate(chunks):
        t0 = time.perf_counter()
        try:
            vector, total_tokens = _get_openai_embedding(content)
            latency_ms = (time.perf_counter() - t0) * 1000
            log_api_call(
                model=EMBED_MODEL,
                stage="rag_embed_chunked",
                input_tokens=total_tokens,
                output_tokens=0,
                latency_ms=latency_ms,
                cost_usd=total_tokens * _EMBED_COST_PER_1M_USD / 1_000_000,
                success=True,
            )
        except Exception as exc:
            LOGGER.warning(
                "Embedding failed for chunk %d, document id=%s: %s",
                idx,
                document.id,
                exc,
            )
            log_api_call(
                model=EMBED_MODEL,
                stage="rag_embed_chunked",
                input_tokens=0,
                output_tokens=0,
                latency_ms=(time.perf_counter() - t0) * 1000,
                cost_usd=0.0,
                success=False,
                error=str(exc),
            )
            continue

        try:
            collection.upsert(
                ids=[f"{document.id}:chunk:{idx}"],
                documents=[content],
                metadatas=[{**metadata, "indexing_version": INDEXING_VERSION}],
                embeddings=[vector],
            )
        except Exception:
            LOGGER.exception(
                "Failed to upsert chunk %d, document id=%s", idx, document.id
            )


def query_chunked(
    question: str, top_k: int = 5, model: str = "gpt-4o-mini", rewrite: bool = False
) -> QAResult:
    """Answer a question using rechunked CatchUp RAG (1000-char chunks, no adjacent expansion).

    Identical control conditions to baseline: same chunk size, same top_k, same LLM.
    Only variable vs baseline: Docling structured parsing vs PyPDF flat extraction.

    Args:
        question: Natural language question.
        top_k: Number of chunks to retrieve.
        model: LLM model identifier. Must be in SUPPORTED_MODELS.
        rewrite: If True, expand the question with LLM query rewriting before embedding.
                 The original question is still sent to the LLM for answer generation.

    Returns:
        QAResult with answer, source_blocks, model name, latency, and token usage.
    """
    if model not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model: {model!r}. Choose from: {SUPPORTED_MODELS}"
        )

    t_total = time.perf_counter()

    collection = _get_rag_collection(RAG_CHUNKED_COLLECTION_NAME)
    if collection is None or collection.count() == 0:
        return QAResult(
            question=question,
            answer="No chunked documents indexed. Run index_document_chunked() first.",
            source_blocks=[],
            model=model,
            latency_ms=(time.perf_counter() - t_total) * 1000,
            input_tokens=0,
            output_tokens=0,
        )

    # Optionally rewrite question for better embedding retrieval
    embed_target = question
    rw_query: str | None = None
    if rewrite:
        embed_target, _rw_lat, _rw_in, _rw_out = _rewrite_query(question)
        rw_query = embed_target
        LOGGER.info("Query rewritten (chunked): %r -> %r", question, embed_target)

    # Embed question
    try:
        t_embed = time.perf_counter()
        question_vector, embed_tokens = _get_openai_embedding(embed_target)
        log_api_call(
            model=EMBED_MODEL,
            stage="rag_embed_chunked",
            input_tokens=embed_tokens,
            output_tokens=0,
            latency_ms=(time.perf_counter() - t_embed) * 1000,
            cost_usd=embed_tokens * _EMBED_COST_PER_1M_USD / 1_000_000,
            success=True,
        )
    except Exception as exc:
        LOGGER.error("Failed to embed question (chunked): %s", exc)
        return QAResult(
            question=question,
            answer="질문 처리에 실패했습니다.",
            source_blocks=[],
            model=model,
            latency_ms=(time.perf_counter() - t_total) * 1000,
            input_tokens=0,
            output_tokens=0,
        )

    # Search
    try:
        n_results = min(top_k, collection.count())
        raw_results = collection.query(
            query_embeddings=[question_vector],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        LOGGER.exception("Chunked ChromaDB search failed")
        raw_results = {}

    ids = (raw_results.get("ids") or [[]])[0]
    contents = (raw_results.get("documents") or [[]])[0]
    metadatas = (raw_results.get("metadatas") or [[]])[0]

    if not ids:
        return QAResult(
            question=question,
            answer="관련 문서를 찾지 못했습니다.",
            source_blocks=[],
            model=model,
            latency_ms=(time.perf_counter() - t_total) * 1000,
            input_tokens=0,
            output_tokens=0,
        )

    source_blocks: list[SourceBlock] = []
    context_parts: list[str] = []
    for i, content in enumerate(contents):
        meta = metadatas[i] if i < len(metadatas) else {}
        page = meta.get("page")
        cell_index = meta.get("cell_index")
        source_blocks.append(
            SourceBlock(
                document_id=meta.get("document_id", ""),
                source=meta.get("source", ""),
                block_order=meta.get("block_order", 0),
                block_type=meta.get("block_type", ""),
                content_preview=content[:200],
                page=page,
                cell_index=cell_index,
                image_path=meta.get("image_path"),
            )
        )
        ref = f"[{meta.get('source', 'unknown')}]"
        if page is not None:
            ref += f" page {page}"
        elif cell_index is not None:
            ref += f" cell {cell_index}"
        context_parts.append(f"{ref}\n{content}")

    context = "\n\n---\n\n".join(context_parts)
    user_content = f"Context:\n{context}\n\nQuestion: {question}"

    try:
        t_llm = time.perf_counter()
        response = call_llm(
            model,
            [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=2048,
        )
        log_api_call(
            model=model,
            stage="rag_generate_chunked",
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=(time.perf_counter() - t_llm) * 1000,
            cost_usd=compute_cost(
                model,
                response.input_tokens,
                response.output_tokens,
            ),
            success=True,
        )
        return QAResult(
            question=question,
            answer=response.content.strip(),
            source_blocks=source_blocks,
            model=model,
            latency_ms=(time.perf_counter() - t_total) * 1000,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            rewritten_query=rw_query,
        )
    except Exception as exc:
        LOGGER.error("Chunked LLM generation failed: %s", exc)
        return QAResult(
            question=question,
            answer="답변 생성에 실패했습니다.",
            source_blocks=source_blocks,
            model=model,
            latency_ms=(time.perf_counter() - t_total) * 1000,
            input_tokens=0,
            output_tokens=0,
        )


def query(
    question: str,
    top_k: int = 5,
    model: str = "gpt-4o-mini",
    document_id: str | None = None,
    rewrite: bool = False,
) -> QAResult:
    """Answer a question using RAG: embed question → search ChromaDB → generate answer with LLM.

    Args:
        question: Natural language question.
        top_k: Number of context blocks to retrieve.
        model: LLM model identifier for answer generation. Must be in SUPPORTED_MODELS.
        document_id: Optional document id to restrict retrieval to a single indexed document.
        rewrite: If True, expand the question with LLM query rewriting before embedding.
                 The original question is still sent to the LLM for answer generation.

    Returns:
        QAResult with answer, source_blocks, model name, latency, and token usage.

    Raises:
        ValueError: If model is not in SUPPORTED_MODELS.
    """
    if model not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model: {model!r}. Choose from: {SUPPORTED_MODELS}"
        )

    t_total = time.perf_counter()

    collection = _get_rag_collection()
    if collection is None or collection.count() == 0:
        return QAResult(
            question=question,
            answer="인덱싱된 문서가 없습니다. 먼저 문서를 인덱싱해주세요. (No documents have been indexed yet.)",
            source_blocks=[],
            model=model,
            latency_ms=(time.perf_counter() - t_total) * 1000,
            input_tokens=0,
            output_tokens=0,
        )

    # Optionally rewrite question for better embedding retrieval
    embed_target = question
    rw_query: str | None = None
    if rewrite:
        embed_target, _rw_lat, _rw_in, _rw_out = _rewrite_query(question)
        rw_query = embed_target
        LOGGER.info("Query rewritten: %r -> %r", question, embed_target)

    # Embed question
    try:
        t_embed = time.perf_counter()
        question_vector, embed_tokens = _get_openai_embedding(embed_target)
        embed_latency_ms = (time.perf_counter() - t_embed) * 1000
        log_api_call(
            model=EMBED_MODEL,
            stage="rag_embed",
            input_tokens=embed_tokens,
            output_tokens=0,
            latency_ms=embed_latency_ms,
            cost_usd=embed_tokens * _EMBED_COST_PER_1M_USD / 1_000_000,
            success=True,
        )
    except Exception as exc:
        LOGGER.error("Failed to embed question: %s", exc)
        log_api_call(
            model=EMBED_MODEL,
            stage="rag_embed",
            input_tokens=0,
            output_tokens=0,
            latency_ms=(time.perf_counter() - t_total) * 1000,
            cost_usd=0.0,
            success=False,
            error=str(exc),
        )
        return QAResult(
            question=question,
            answer="질문 처리에 실패했습니다. 잠시 후 다시 시도해주세요.",
            source_blocks=[],
            model=model,
            latency_ms=(time.perf_counter() - t_total) * 1000,
            input_tokens=0,
            output_tokens=0,
        )

    # Search ChromaDB
    try:
        if document_id is not None:
            filtered_count = len(
                collection.get(where={"document_id": {"$eq": document_id}})["ids"]
            )
            n_results = min(top_k, filtered_count)
        else:
            n_results = min(top_k, collection.count())
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [question_vector],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if document_id is not None:
            query_kwargs["where"] = {"document_id": {"$eq": document_id}}
        raw_results = collection.query(
            **query_kwargs,
        )
    except Exception:
        LOGGER.exception("ChromaDB search failed")
        raw_results = {}

    ids = (raw_results.get("ids") or [[]])[0]
    contents = (raw_results.get("documents") or [[]])[0]
    metadatas = (raw_results.get("metadatas") or [[]])[0]

    if not ids:
        return QAResult(
            question=question,
            answer="관련 문서를 찾지 못했습니다.",
            source_blocks=[],
            model=model,
            latency_ms=(time.perf_counter() - t_total) * 1000,
            input_tokens=0,
            output_tokens=0,
        )

    # Expand hits with adjacent blocks for wider context coverage
    expanded = _expand_with_adjacent_blocks(collection, metadatas, contents)

    # Build source blocks (original top-k hits only) and context string (expanded)
    hit_keys = {
        f"{meta.get('document_id', '')}:{meta.get('block_order', 0)}"
        for meta in metadatas
    }
    source_blocks: list[SourceBlock] = []
    context_parts: list[str] = []

    for meta, content in expanded:
        key = f"{meta.get('document_id', '')}:{meta.get('block_order', 0)}"
        page = meta.get("page")
        cell_index = meta.get("cell_index")

        # source_blocks tracks original retrieved hits only
        if key in hit_keys:
            source_blocks.append(
                SourceBlock(
                    document_id=meta.get("document_id", ""),
                    source=meta.get("source", ""),
                    block_order=meta.get("block_order", 0),
                    block_type=meta.get("block_type", ""),
                    content_preview=content[:200],
                    page=page,
                    cell_index=cell_index,
                    image_path=meta.get("image_path"),
                )
            )

        ref = f"[{meta.get('source', 'unknown')}]"
        if page is not None:
            ref += f" page {page}"
        elif cell_index is not None:
            ref += f" cell {cell_index}"
        context_parts.append(f"{ref}\n{content}")

    context = "\n\n---\n\n".join(context_parts)
    user_content = f"Context:\n{context}\n\nQuestion: {question}"

    # LLM answer generation
    try:
        t_llm = time.perf_counter()
        response = call_llm(
            model,
            [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=2048,
        )
        llm_latency_ms = (time.perf_counter() - t_llm) * 1000
        cost_usd = compute_cost(
            model,
            response.input_tokens,
            response.output_tokens,
        )

        log_api_call(
            model=model,
            stage="rag_generate",
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=llm_latency_ms,
            cost_usd=cost_usd,
            success=True,
        )
        return QAResult(
            question=question,
            answer=response.content.strip(),
            source_blocks=source_blocks,
            model=model,
            latency_ms=(time.perf_counter() - t_total) * 1000,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            rewritten_query=rw_query,
        )

    except Exception as exc:
        LOGGER.error("LLM answer generation failed: %s", exc)
        log_api_call(
            model=model,
            stage="rag_generate",
            input_tokens=0,
            output_tokens=0,
            latency_ms=(time.perf_counter() - t_total) * 1000,
            cost_usd=0.0,
            success=False,
            error=str(exc),
        )
        return QAResult(
            question=question,
            answer="답변 생성에 실패했습니다. 잠시 후 다시 시도해주세요.",
            source_blocks=source_blocks,
            model=model,
            latency_ms=(time.perf_counter() - t_total) * 1000,
            input_tokens=0,
            output_tokens=0,
        )


def retrieve_context(
    query_text: str,
    document_id: str,
    top_k: int = 5,
) -> list[str]:
    """Retrieve top_k content chunks from ChromaDB scoped to a single document.

    Embeds query_text and returns the matching chunk strings. Used to ground
    note edits in actual document content rather than LLM parametric knowledge.

    Returns an empty list on any error (network, ChromaDB, or embedding failure).

    Args:
        query_text: The edit instruction or keyword string to embed.
        document_id: Restrict retrieval to this document's indexed chunks.
        top_k: Maximum number of chunks to return.
    """
    collection = _get_rag_collection()
    if collection is None:
        return []

    try:
        t0 = time.perf_counter()
        query_vector, embed_tokens = _get_openai_embedding(query_text)
        latency_ms = (time.perf_counter() - t0) * 1000
        log_api_call(
            model=EMBED_MODEL,
            stage="note_edit_retrieve",
            input_tokens=embed_tokens,
            output_tokens=0,
            latency_ms=latency_ms,
            cost_usd=embed_tokens * _EMBED_COST_PER_1M_USD / 1_000_000,
            success=True,
            error=None,
        )
    except Exception:
        LOGGER.exception("Failed to embed query for context retrieval")
        return []

    try:
        filtered = collection.get(where={"document_id": {"$eq": document_id}})
        filtered_count = len(filtered.get("ids") or [])
        if filtered_count == 0:
            return []
        n_results = min(top_k, filtered_count)
        raw = collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            include=["documents"],
            where={"document_id": {"$eq": document_id}},
        )
        return (raw.get("documents") or [[]])[0]
    except Exception:
        LOGGER.exception(
            "ChromaDB context retrieval failed for document_id=%s", document_id
        )
        return []


def has_document_vectors(document_id: str) -> bool:
    """Return True if at least one vector is stored for this document in the RAG collection.

    Used by the UI to decide whether to set the indexed session-state flag when restoring
    a document from the library — avoids marking a document as indexed when embeddings were
    never stored or were deleted.

    Args:
        document_id: Document.id to check.
    """
    collection = _get_rag_collection()
    if collection is None:
        return False
    try:
        result = collection.get(where={"document_id": {"$eq": document_id}})
        return len(result.get("ids") or []) > 0
    except Exception:
        return False


def delete_document_index(document_id: str) -> None:
    """Delete all ChromaDB entries for a document from both RAG collections.

    Called when a document is removed from the library so stale vectors don't
    persist and re-uploads trigger a fresh index with current filters applied.

    Args:
        document_id: Document.id whose entries should be removed.
    """
    for name in (RAG_COLLECTION_NAME, RAG_CHUNKED_COLLECTION_NAME):
        collection = _get_rag_collection(name)
        if collection is None:
            continue
        try:
            result = collection.get(where={"document_id": document_id})
            ids = result.get("ids", [])
            if ids:
                collection.delete(ids=ids)
                LOGGER.info(
                    "Deleted %d vectors for document id=%s from %s",
                    len(ids),
                    document_id,
                    name,
                )
        except Exception as exc:
            LOGGER.warning(
                "Failed to delete vectors for document id=%s from %s: %s",
                document_id,
                name,
                exc,
            )


__all__ = [
    "index_document",
    "index_document_chunked",
    "delete_document_index",
    "has_document_vectors",
    "query",
    "query_chunked",
    "retrieve_context",
    "rechunk_blocks",
    "QAResult",
    "SourceBlock",
    "SUPPORTED_MODELS",
    "RAG_COLLECTION_NAME",
    "RAG_CHUNKED_COLLECTION_NAME",
    "INDEXING_VERSION",
]

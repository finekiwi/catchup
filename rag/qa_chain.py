"""RAG Q&A pipeline: index Document blocks into ChromaDB and answer questions with LLM.

Workflow:
  1. index_document() embeds each block with text-embedding-3-small and stores in ChromaDB.
  2. query() embeds the question, retrieves top-k blocks, and generates an answer with the LLM.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from db.chroma import _build_client
from models.document import Document
from prompts.rag_qa import PROMPT
from utils.logging import log_api_call

load_dotenv()

LOGGER = logging.getLogger(__name__)

RAG_COLLECTION_NAME = "catchup_rag"
EMBED_MODEL = "text-embedding-3-small"
_EMBED_COST_PER_1M_USD = 0.02  # USD per 1M tokens for text-embedding-3-small

# ---------------------------------------------------------------------------
# Model registry — mirrors note_generator.py
# ---------------------------------------------------------------------------
_MODEL_REGISTRY: dict[str, dict] = {
    "gpt-4o-mini":                   {"provider": "openai",    "input": 0.15,  "output": 0.60},
    "gpt-4o":                        {"provider": "openai",    "input": 2.50,  "output": 10.00},
    "gpt-4.1-mini":                  {"provider": "openai",    "input": 0.40,  "output": 1.60},
    "gpt-4.1-nano":                  {"provider": "openai",    "input": 0.10,  "output": 0.40},
    "gpt-5-nano":                    {"provider": "openai",    "input": 0.20,  "output": 0.80},
    "claude-haiku-4-5-20251001":     {"provider": "anthropic", "input": 0.80,  "output": 4.00},
    "claude-sonnet-4-6":             {"provider": "anthropic", "input": 3.00,  "output": 15.00},
    "gemini-3-flash-preview":        {"provider": "google",    "input": 0.10,  "output": 0.40},
    "gemini-3.1-pro-preview":        {"provider": "google",    "input": 1.25,  "output": 10.00},
    "gemini-3.1-flash-lite-preview": {"provider": "google",    "input": 0.04,  "output": 0.15},
}

SUPPORTED_MODELS: list[str] = list(_MODEL_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

class SourceBlock(BaseModel):
    """A retrieved document block used as evidence for an answer."""

    document_id: str
    source: str            # original filename
    block_order: int
    block_type: str
    content_preview: str   # first 200 chars
    page: Optional[int] = None
    cell_index: Optional[int] = None


class QAResult(BaseModel):
    """Result of a RAG Q&A query."""

    question: str
    answer: str
    source_blocks: list[SourceBlock] = Field(default_factory=list)
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD from token counts."""
    info = _MODEL_REGISTRY.get(model, {})
    return (
        input_tokens * info.get("input", 0) / 1_000_000
        + output_tokens * info.get("output", 0) / 1_000_000
    )


def _get_rag_collection() -> Optional[Any]:
    """Get or create the RAG ChromaDB collection."""
    try:
        client = _build_client()
        return client.get_or_create_collection(name=RAG_COLLECTION_NAME)
    except Exception:
        LOGGER.exception("Failed to initialize RAG ChromaDB collection")
        return None


def _get_openai_embedding(text: str) -> tuple[list[float], int]:
    """Embed text using OpenAI text-embedding-3-small. Returns (vector, total_tokens)."""
    import openai  # lazy import

    client = openai.OpenAI()
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding, resp.usage.total_tokens


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
                where={"$and": [
                    {"document_id": {"$eq": doc_id}},
                    {"block_order": {"$in": sorted(orders)}},
                ]},
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
    expanded.sort(key=lambda x: (x[0].get("document_id", ""), int(x[0].get("block_order", 0))))
    return expanded


def _is_document_indexed(collection: Any, document_id: str, expected_block_count: int) -> bool:
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


# ---------------------------------------------------------------------------
# Provider-level LLM call functions — mirrors note_generator.py
# ---------------------------------------------------------------------------

def _call_openai(model: str, system: str, user: str) -> tuple[str, int, int]:
    """Call OpenAI chat completion API."""
    import openai  # lazy import

    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=2048,
    )
    raw = resp.choices[0].message.content or ""
    return raw, resp.usage.prompt_tokens, resp.usage.completion_tokens


def _call_anthropic(model: str, system: str, user: str) -> tuple[str, int, int]:
    """Call Anthropic Claude chat API."""
    import anthropic  # lazy import

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = resp.content[0].text if resp.content else ""
    return raw, resp.usage.input_tokens, resp.usage.output_tokens


def _call_google(model: str, system: str, user: str) -> tuple[str, int, int]:
    """Call Google Gemini chat API."""
    import google.generativeai as genai  # lazy import

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    gemini = genai.GenerativeModel(model, system_instruction=system)
    resp = gemini.generate_content(user)
    raw = resp.text or ""
    meta = getattr(resp, "usage_metadata", None)
    input_tokens = getattr(meta, "prompt_token_count", 0) or 0
    output_tokens = getattr(meta, "candidates_token_count", 0) or 0
    return raw, input_tokens, output_tokens


_PROVIDER_DISPATCH: dict[str, Callable[..., tuple[str, int, int]]] = {
    "openai":    _call_openai,
    "anthropic": _call_anthropic,
    "google":    _call_google,
}


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
        LOGGER.error("RAG collection unavailable — skipping index for document id=%s", document.id)
        return

    non_empty_blocks = [b for b in document.blocks if b.content.strip()]
    if _is_document_indexed(collection, document.id, len(non_empty_blocks)):
        LOGGER.info("Document id=%s already indexed, skipping", document.id)
        return

    for block in document.blocks:
        content = block.content.strip()
        if not content:
            continue

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
                block.order, document.id, exc,
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
        }
        if block.metadata.page is not None:
            metadata["page"] = block.metadata.page
        if block.metadata.cell_index is not None:
            metadata["cell_index"] = block.metadata.cell_index

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
                block.order, document.id,
            )


def query(question: str, top_k: int = 5, model: str = "gpt-4o-mini") -> QAResult:
    """Answer a question using RAG: embed question → search ChromaDB → generate answer with LLM.

    Args:
        question: Natural language question.
        top_k: Number of context blocks to retrieve.
        model: LLM model identifier for answer generation. Must be in SUPPORTED_MODELS.

    Returns:
        QAResult with answer, source_blocks, model name, latency, and token usage.

    Raises:
        ValueError: If model is not in SUPPORTED_MODELS.
    """
    if model not in _MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {model!r}. Choose from: {SUPPORTED_MODELS}")

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

    # Embed question
    try:
        t_embed = time.perf_counter()
        question_vector, embed_tokens = _get_openai_embedding(question)
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
        n_results = min(top_k, collection.count())
        raw_results = collection.query(
            query_embeddings=[question_vector],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
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
            source_blocks.append(SourceBlock(
                document_id=meta.get("document_id", ""),
                source=meta.get("source", ""),
                block_order=meta.get("block_order", 0),
                block_type=meta.get("block_type", ""),
                content_preview=content[:200],
                page=page,
                cell_index=cell_index,
            ))

        ref = f"[{meta.get('source', 'unknown')}]"
        if page is not None:
            ref += f" page {page}"
        elif cell_index is not None:
            ref += f" cell {cell_index}"
        context_parts.append(f"{ref}\n{content}")

    context = "\n\n---\n\n".join(context_parts)
    user_content = f"Context:\n{context}\n\nQuestion: {question}"

    # LLM answer generation
    provider = _MODEL_REGISTRY.get(model, {}).get("provider", "openai")
    call_fn = _PROVIDER_DISPATCH.get(provider)

    if call_fn is None:
        LOGGER.error("No provider dispatch found for model=%s", model)
        return QAResult(
            question=question,
            answer="지원하지 않는 모델입니다.",
            source_blocks=source_blocks,
            model=model,
            latency_ms=(time.perf_counter() - t_total) * 1000,
            input_tokens=0,
            output_tokens=0,
        )

    try:
        t_llm = time.perf_counter()
        raw_answer, input_tokens, output_tokens = call_fn(model, PROMPT, user_content)
        llm_latency_ms = (time.perf_counter() - t_llm) * 1000
        cost_usd = _compute_cost(model, input_tokens, output_tokens)

        log_api_call(
            model=model,
            stage="rag_generate",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=llm_latency_ms,
            cost_usd=cost_usd,
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


__all__ = ["index_document", "query", "QAResult", "SourceBlock", "SUPPORTED_MODELS"]

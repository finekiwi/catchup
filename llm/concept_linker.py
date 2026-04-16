"""Concept linking pipeline: normalize → embed → search → label."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from dotenv import load_dotenv

from db.sqlite import (
    delete_concepts_for_document,
    get_all_concepts,
    save_concept_links,
    save_concepts,
)
from prompts.concept_linking import (
    CANONICAL_NORMALIZE_PROMPT,
    RELATIONSHIP_LABEL_PROMPT,
    get_label_prompt,
    get_normalize_prompt,
)
from utils.embed import get_openai_embedding as _get_openai_embedding
from utils.logging import log_api_call

load_dotenv()

LOGGER = logging.getLogger(__name__)

CONCEPTS_COLLECTION_NAME = "catchup_concepts"
_EMBED_MODEL = "text-embedding-3-small"
_EMBED_COST_PER_1M_USD = 0.02
_DEFAULT_NORMALIZE_MODEL = "gpt-4.1-nano"

# Regex to strip markdown code fences from LLM responses
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_fence(text: str) -> str:
    """Remove markdown code fences from an LLM text response."""
    return _FENCE_RE.sub("", text).strip()


def _get_concepts_collection() -> Any | None:
    """Get or create the catchup_concepts ChromaDB collection with cosine distance space."""
    try:
        from db.chroma import _build_client

        client = _build_client()
        return client.get_or_create_collection(
            name=CONCEPTS_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception:
        LOGGER.exception("Failed to initialize concepts ChromaDB collection")
        return None


def normalize_concepts(raw_concepts: list[str], model: str) -> list[dict]:
    """Call LLM once to get canonical name, aliases, and definition for each concept.

    Uses gpt-4.1-nano when available, otherwise falls back to the provided model.
    Logs the API call with stage='concept_normalize'.

    Args:
        raw_concepts: List of raw concept name strings from note generation.
        model: Fallback LLM model if gpt-4.1-nano is unavailable.

    Returns:
        List of dicts with keys: raw, canonical, aliases, definition.
        Empty list if raw_concepts is empty or API call fails.
    """
    if not raw_concepts:
        return []

    # Prefer cheap nano model for normalization
    normalize_model = _DEFAULT_NORMALIZE_MODEL

    import openai  # lazy import

    client = openai.OpenAI()
    user_content = get_normalize_prompt(raw_concepts)

    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=normalize_model,
            messages=[
                {"role": "system", "content": CANONICAL_NORMALIZE_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        raw_text = resp.choices[0].message.content or ""
        input_tokens = resp.usage.prompt_tokens
        output_tokens = resp.usage.completion_tokens
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        LOGGER.warning("normalize_concepts API call failed: %s", exc)
        log_api_call(
            model=normalize_model,
            stage="concept_normalize",
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            cost_usd=0.0,
            success=False,
            error=str(exc),
        )
        return []

    log_api_call(
        model=normalize_model,
        stage="concept_normalize",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        cost_usd=0.0,
        success=True,
    )

    # Parse JSON — LLM returns {"concepts": [...]} wrapper (json_object mode forbids top-level arrays)
    try:
        cleaned = _strip_fence(raw_text)
        parsed = json.loads(cleaned)
        # Unwrap {"concepts": [...]} or any dict with a single list value
        if isinstance(parsed, dict):
            if "concepts" in parsed and isinstance(parsed["concepts"], list):
                parsed = parsed["concepts"]
            else:
                # Fallback: find first list value
                for v in parsed.values():
                    if isinstance(v, list):
                        parsed = v
                        break
                else:
                    LOGGER.warning("normalize_concepts: unexpected dict response (no list value)")
                    return []
        if not isinstance(parsed, list):
            LOGGER.warning("normalize_concepts: JSON is not a list, returning empty")
            return []
        results: list[dict] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            results.append(
                {
                    "raw": item.get("raw", ""),
                    "canonical": (item.get("canonical") or "").lower().strip(),
                    "aliases": item.get("aliases") or [],
                    "definition": item.get("definition") or "",
                }
            )
        return results
    except (json.JSONDecodeError, ValueError) as exc:
        LOGGER.warning("normalize_concepts JSON parse failed: %s", exc)
        return []


def embed_and_store_concepts(document_id: str, concepts: list[dict]) -> None:
    """Embed each concept and upsert into the catchup_concepts ChromaDB collection.

    Embedding input format: "{canonical} ({aliases}): {definition}"
    Logs each embedding call with stage='concept_embed'.

    Args:
        document_id: Document.id owning these concepts.
        concepts: List of concept dicts (output of normalize_concepts + id from save_concepts).
    """
    if not concepts:
        return

    collection = _get_concepts_collection()
    if collection is None:
        LOGGER.error("concepts collection unavailable — skipping embed for document_id=%s", document_id)
        return

    for idx, concept in enumerate(concepts):
        canonical = concept.get("canonical_name") or concept.get("canonical") or ""
        aliases = concept.get("aliases") or []
        definition = concept.get("definition") or ""
        aliases_str = ", ".join(aliases) if aliases else canonical
        embed_text = f"{canonical} ({aliases_str}): {definition}"

        t0 = time.perf_counter()
        try:
            vector, total_tokens = _get_openai_embedding(embed_text)
            latency_ms = (time.perf_counter() - t0) * 1000
            log_api_call(
                model=_EMBED_MODEL,
                stage="concept_embed",
                input_tokens=total_tokens,
                output_tokens=0,
                latency_ms=latency_ms,
                cost_usd=total_tokens * _EMBED_COST_PER_1M_USD / 1_000_000,
                success=True,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            LOGGER.warning("concept embedding failed for idx=%d document_id=%s: %s", idx, document_id, exc)
            log_api_call(
                model=_EMBED_MODEL,
                stage="concept_embed",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                cost_usd=0.0,
                success=False,
                error=str(exc),
            )
            continue

        metadata: dict[str, Any] = {
            "document_id": document_id,
            "concept_name": concept.get("concept_name") or concept.get("raw") or canonical,
            "canonical_name": canonical,
            "aliases": json.dumps(aliases, ensure_ascii=False),
            "definition": definition,
        }
        chroma_id = f"{document_id}:{idx}"
        try:
            collection.upsert(
                ids=[chroma_id],
                documents=[embed_text],
                metadatas=[metadata],
                embeddings=[vector],
            )
        except Exception:
            LOGGER.exception("Failed to upsert concept idx=%d document_id=%s", idx, document_id)


def find_exact_matches(document_id: str, concepts: list[dict]) -> list[dict]:
    """Tier 1: Find concepts from other documents with identical canonical name.

    Exact canonical match → confidence=1.0, relationship_type="same_concept".
    No LLM call required.

    Args:
        document_id: Document.id of the newly uploaded document.
        concepts: List of normalized concept dicts for the new document
                  (must include 'id' field from save_concepts).

    Returns:
        List of match dicts with keys: concept_a, concept_b, confidence_score,
        relationship_type, relationship_desc, concept_id_a, concept_id_b.
    """
    existing = get_all_concepts(exclude_document_id=document_id)
    if not existing:
        return []

    # Build lookup: canonical_name → list of existing concepts
    existing_by_canonical: dict[str, list[dict]] = {}
    for ex in existing:
        key = (ex.get("canonical_name") or "").lower()
        existing_by_canonical.setdefault(key, []).append(ex)

    matches: list[dict] = []
    for concept in concepts:
        canonical = (concept.get("canonical_name") or "").lower()
        if not canonical:
            continue
        for other in existing_by_canonical.get(canonical, []):
            matches.append(
                {
                    "concept_a": concept,
                    "concept_b": other,
                    "concept_id_a": concept["id"],
                    "concept_id_b": other["id"],
                    "confidence_score": 1.0,
                    "relationship_type": "same_concept",
                    "relationship_desc": "",
                }
            )
    return matches


def find_similar_concepts(
    document_id: str,
    concepts: list[dict],
    already_matched_pairs: set[tuple],
    threshold: float = 0.75,
    top_k: int = 3,
) -> list[dict]:
    """Tier 2: Find similar concepts via ChromaDB cosine similarity.

    Excludes same-document concepts and pairs already matched by Tier 1.
    Returns up to top_k pairs with confidence_score = cosine similarity.

    Args:
        document_id: Document.id of the newly uploaded document.
        concepts: Normalized concept dicts for the new document (with 'id' field).
        already_matched_pairs: Set of (min_id, max_id) pairs already found in Tier 1.
        threshold: Minimum cosine similarity to consider a match (default 0.75).
        top_k: Maximum number of pairs to return across all queries.

    Returns:
        List of similarity match dicts, each with concept_a, concept_b,
        concept_id_a, concept_id_b, confidence_score.
    """
    collection = _get_concepts_collection()
    if collection is None:
        return []

    try:
        count = collection.count()
    except Exception:
        count = 0

    if count == 0:
        return []

    pairs: list[dict] = []
    seen_pairs: set[tuple] = set(already_matched_pairs)

    for concept in concepts:
        canonical = concept.get("canonical_name") or concept.get("canonical") or ""
        aliases = concept.get("aliases") or []
        definition = concept.get("definition") or ""
        aliases_str = ", ".join(aliases) if aliases else canonical
        embed_text = f"{canonical} ({aliases_str}): {definition}"

        try:
            vector, _ = _get_openai_embedding(embed_text)
        except Exception as exc:
            LOGGER.warning("embed failed for concept %r: %s", canonical, exc)
            continue

        try:
            # Exclude same-document concepts at query time so self-hits never consume result slots
            n_results = min(10, count)
            raw = collection.query(
                query_embeddings=[vector],
                n_results=n_results,
                where={"document_id": {"$ne": document_id}},
                include=["metadatas", "distances"],
            )
        except Exception:
            LOGGER.exception("ChromaDB query failed for concept %r", canonical)
            continue

        hit_metas: list[dict] = (raw.get("metadatas") or [[]])[0]
        hit_distances: list[float] = (raw.get("distances") or [[]])[0]

        for meta, distance in zip(hit_metas, hit_distances):
            # ChromaDB with cosine space returns distance = 1 - cosine_sim
            similarity = 1.0 - distance

            if similarity < threshold:
                continue

            hit_doc_id = meta.get("document_id", "")

            # Resolve the SQLite concept_id from the ChromaDB hit
            # The concept_id is NOT stored in ChromaDB metadata; we look it up from SQLite
            other_concepts = get_all_concepts(exclude_document_id=document_id)
            hit_canonical = meta.get("canonical_name", "")
            hit_concept_name = meta.get("concept_name", "")
            other_concept: dict | None = None
            for oc in other_concepts:
                if oc["document_id"] == hit_doc_id and (
                    oc["canonical_name"] == hit_canonical or oc["concept_name"] == hit_concept_name
                ):
                    other_concept = oc
                    break

            if other_concept is None:
                continue

            concept_id_a = concept["id"]
            concept_id_b = other_concept["id"]
            pair_key = (min(concept_id_a, concept_id_b), max(concept_id_a, concept_id_b))
            if pair_key in seen_pairs:
                continue

            seen_pairs.add(pair_key)
            pairs.append(
                {
                    "concept_a": concept,
                    "concept_b": other_concept,
                    "concept_id_a": concept_id_a,
                    "concept_id_b": concept_id_b,
                    "confidence_score": similarity,
                    "relationship_type": "",
                    "relationship_desc": "",
                }
            )

            if len(pairs) >= top_k:
                return pairs

    return pairs[:top_k]


def label_relationships(pairs: list[dict], model: str) -> list[dict]:
    """Tier 2 only: label the semantic relationship for each candidate pair via LLM.

    Drops pairs where the LLM returns relationship_type=null (precision over recall).
    Logs each API call with stage='concept_label'.

    Args:
        pairs: Candidate pairs from find_similar_concepts.
        model: LLM model to use for labeling.

    Returns:
        Subset of pairs that received a valid (non-null) relationship_type.
    """
    if not pairs:
        return []

    import openai  # lazy import

    client = openai.OpenAI()
    labeled: list[dict] = []

    for pair in pairs:
        concept_a = pair["concept_a"]
        concept_b = pair["concept_b"]

        source_info = {
            "canonical_name": concept_a.get("canonical_name") or concept_a.get("canonical", ""),
            "aliases": concept_a.get("aliases") or [],
            "definition": concept_a.get("definition") or "",
            "doc_title": concept_a.get("document_id", "?"),
        }
        target_info = {
            "canonical_name": concept_b.get("canonical_name") or concept_b.get("canonical", ""),
            "aliases": concept_b.get("aliases") or [],
            "definition": concept_b.get("definition") or "",
            "doc_title": concept_b.get("document_id", "?"),
        }
        user_content = get_label_prompt(source_info, target_info)

        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": RELATIONSHIP_LABEL_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=128,
                response_format={"type": "json_object"},
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            raw_text = resp.choices[0].message.content or ""
            input_tokens = resp.usage.prompt_tokens
            output_tokens = resp.usage.completion_tokens
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            LOGGER.warning("label_relationships API call failed: %s", exc)
            log_api_call(
                model=model,
                stage="concept_label",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                cost_usd=0.0,
                success=False,
                error=str(exc),
            )
            continue

        log_api_call(
            model=model,
            stage="concept_label",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=0.0,
            success=True,
        )

        try:
            parsed = json.loads(_strip_fence(raw_text))
        except (json.JSONDecodeError, ValueError) as exc:
            LOGGER.warning("label_relationships JSON parse failed: %s", exc)
            continue

        rel_type = parsed.get("relationship_type")
        if rel_type is None:
            # Drop pair — precision over recall
            continue

        valid_types = {"implements", "extends", "prerequisite", "application"}
        if rel_type not in valid_types:
            LOGGER.debug("label_relationships: unknown type %r — dropping", rel_type)
            continue

        labeled_pair = dict(pair)
        labeled_pair["relationship_type"] = rel_type
        labeled_pair["relationship_desc"] = parsed.get("description") or ""
        labeled.append(labeled_pair)

    return labeled


def link_concepts(document_id: str, key_concepts: list[str], model: str, threshold: float = 0.75) -> list[dict]:
    """Main entry point: run full concept linking pipeline for a newly indexed document.

    Pipeline:
    1. Normalize raw concept names via LLM (canonical + aliases + definition).
    2. Probe-save new concepts to SQLite — confirm write succeeds before touching existing data.
    3. Delete existing concept data (links + concepts + ChromaDB vectors) — safe because step 2 confirmed SQLite is healthy.
    4. Re-save normalized concepts to SQLite to get authoritative integer IDs after the clean delete.
    5. Embed and store concepts in ChromaDB catchup_concepts collection.
    6. Tier 1: exact canonical match search.
    7. Tier 2: ChromaDB similarity search (threshold=0.75, top_k=3).
    8. LLM relationship labeling for Tier 2 pairs only.
    9. Combine Tier 1 + labeled Tier 2, save links to SQLite.
    10. Return connection list for UI rendering.

    Args:
        document_id: Document.id of the newly uploaded document.
        key_concepts: Raw concept name strings from note generation output.
        model: LLM model for normalization and labeling.
        threshold: Cosine similarity threshold for Tier 2 search (default 0.80).

    Returns:
        List of connection dicts suitable for _render_concept_connections() in UI.
        Each dict matches the structure returned by get_concept_links_for_document().
    """
    if not key_concepts:
        return []

    # Step 1: normalize (fallible — do NOT delete existing data until this succeeds)
    normalized = normalize_concepts(key_concepts, model)
    if not normalized:
        LOGGER.warning("link_concepts: normalization returned empty for document_id=%s", document_id)
        return []

    # Build concept dicts for saving — use raw as concept_name, canonical as canonical_name
    concept_rows = [
        {
            "concept_name": c.get("raw") or c.get("canonical") or "",
            "canonical_name": c.get("canonical") or "",
            "aliases": c.get("aliases") or [],
            "definition": c.get("definition") or "",
        }
        for c in normalized
    ]

    # Step 2: save to SQLite first — confirm the write succeeds before touching existing data.
    # Attempting save_concepts before delete ensures we never wipe valid existing concepts
    # and then fail to insert replacements (atomic-style: verify write, then clean, then finalize).
    probe_ids = save_concepts(document_id, concept_rows)
    if not probe_ids:
        LOGGER.warning("link_concepts: save_concepts probe returned empty IDs for document_id=%s", document_id)
        return []

    # Step 3: delete old concept data (links + concepts + ChromaDB vectors) now that we know
    # the SQLite write path is healthy.  delete_document_concepts also removes the rows we
    # just inserted above, so we re-save below to get the final clean IDs.
    delete_document_concepts(document_id)

    # Step 4: save final concepts to SQLite, get authoritative integer IDs.
    ids = save_concepts(document_id, concept_rows)
    if not ids:
        LOGGER.warning("link_concepts: save_concepts returned empty IDs for document_id=%s", document_id)
        return []

    # Step 5: embed and store in ChromaDB (after SQLite commit is confirmed)
    embed_and_store_concepts(document_id, concept_rows)

    # Attach IDs to concept_rows for pair matching
    concepts_with_ids = []
    for idx, (row, cid) in enumerate(zip(concept_rows, ids)):
        enriched = dict(row)
        enriched["id"] = cid
        enriched["aliases"] = (normalized[idx].get("aliases") or []) if idx < len(normalized) else []
        concepts_with_ids.append(enriched)

    # Step 6: Tier 1 — exact canonical match
    tier1_pairs = find_exact_matches(document_id, concepts_with_ids)
    tier1_pair_keys: set[tuple] = {
        (min(p["concept_id_a"], p["concept_id_b"]), max(p["concept_id_a"], p["concept_id_b"]))
        for p in tier1_pairs
    }

    # Step 7: Tier 2 — similarity search
    tier2_candidates = find_similar_concepts(
        document_id,
        concepts_with_ids,
        already_matched_pairs=tier1_pair_keys,
        threshold=threshold,
        top_k=3,
    )

    # Step 8: LLM labeling for Tier 2
    tier2_labeled = label_relationships(tier2_candidates, model)

    # Step 9: combine and save
    all_pairs = tier1_pairs + tier2_labeled
    if all_pairs:
        link_rows = [
            {
                "concept_id_a": p["concept_id_a"],
                "concept_id_b": p["concept_id_b"],
                "confidence_score": p["confidence_score"],
                "relationship_type": p.get("relationship_type") or "",
                "relationship_desc": p.get("relationship_desc") or "",
            }
            for p in all_pairs
        ]
        save_concept_links(link_rows)

    # Step 10: return UI-ready connections from SQLite join
    from db.sqlite import get_concept_links_for_document

    return get_concept_links_for_document(document_id)


def delete_document_concepts(document_id: str) -> None:
    """Idempotently delete concept links, concepts (SQLite), and vectors (ChromaDB).

    Safe to call even if no concepts exist for the document.

    Args:
        document_id: Document.id whose concept data should be removed.
    """
    # Delete from SQLite (links → concepts, cascading)
    delete_concepts_for_document(document_id)

    # Delete from ChromaDB catchup_concepts collection
    collection = _get_concepts_collection()
    if collection is None:
        return
    try:
        result = collection.get(where={"document_id": document_id})
        ids = result.get("ids") or []
        if ids:
            collection.delete(ids=ids)
            LOGGER.info(
                "Deleted %d concept vectors for document_id=%s from %s",
                len(ids),
                document_id,
                CONCEPTS_COLLECTION_NAME,
            )
    except Exception as exc:
        LOGGER.warning(
            "Failed to delete concept vectors for document_id=%s: %s", document_id, exc
        )

"""PDF chunking size sweep — DeepEval 4 metrics across multiple max_tokens values.

Indexes golden PDFs into separate ChromaDB collections per token size and evaluates
each with the full DeepEval suite. Merges with existing Round 4/5/6 results
(300/400/500 tokens) to produce a complete sweep report.

Usage:
    python -m eval.run_chunking_sweep
    python -m eval.run_chunking_sweep --tokens 750 1000 --skip-existing

Output:
    eval/results/pdf_chunking_sweep.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
LOGGER = logging.getLogger(__name__)

_GOLDEN_DIR = Path("data/golden")
_GOLDEN_PATH = Path("eval/golden_set.json")
_OUTPUT_PATH = Path("eval/results/pdf_chunking_sweep.json")
_DEFAULT_TOKENS = [750, 1000]

# Existing round results to merge in (token_size → file path)
_EXISTING_RESULTS: dict[int, Path] = {
    300: Path("data/eval_results/deepeval_catchup-chunked_round4_20260312T081720.json"),
    400: Path("data/eval_results/deepeval_catchup-chunked_round6_20260312T124822.json"),
    500: Path("data/eval_results/deepeval_catchup-chunked_round5_20260312T084747.json"),
}


# ---------------------------------------------------------------------------
# Per-size indexing helpers (self-contained, no modifications to qa_chain.py)
# ---------------------------------------------------------------------------


def _collection_name(max_tokens: int) -> str:
    return f"catchup_rag_sweep_{max_tokens}"


def _get_collection(name: str) -> Optional[Any]:
    from db.chroma import _build_client

    try:
        return _build_client().get_or_create_collection(name=name)
    except Exception:
        LOGGER.exception("Failed to get collection: %s", name)
        return None


def _index_pdfs_for_tokens(max_tokens: int) -> None:
    """Index all golden PDFs into a dedicated collection for this token size."""
    from parsers.pdf_parser import parse_pdf
    from rag.qa_chain import rechunk_blocks, _get_openai_embedding, EMBED_MODEL
    from utils.logging import log_api_call

    coll_name = _collection_name(max_tokens)
    collection = _get_collection(coll_name)
    if collection is None:
        return

    pdfs = sorted(_GOLDEN_DIR.glob("*.pdf"))
    LOGGER.info(
        "Indexing %d PDFs → %s (max_tokens=%d)", len(pdfs), coll_name, max_tokens
    )

    for pdf in pdfs:
        doc = parse_pdf(str(pdf))

        # Skip if already indexed
        try:
            existing = collection.get(where={"document_id": doc.id})
            if existing.get("ids"):
                LOGGER.info("  %s already indexed in %s, skipping", pdf.name, coll_name)
                continue
        except Exception:
            pass

        chunks = rechunk_blocks(doc, max_tokens=max_tokens)
        LOGGER.info(
            "  %s → %d chunks (max_tokens=%d)", pdf.name, len(chunks), max_tokens
        )

        for idx, (content, metadata) in enumerate(chunks):
            t0 = time.perf_counter()
            try:
                vector, total_tokens = _get_openai_embedding(content)
                log_api_call(
                    model=EMBED_MODEL,
                    stage=f"sweep_embed_{max_tokens}",
                    input_tokens=total_tokens,
                    output_tokens=0,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    cost_usd=total_tokens * 0.02 / 1_000_000,
                    success=True,
                )
            except Exception as exc:
                LOGGER.warning("Embed failed chunk %d of %s: %s", idx, pdf.name, exc)
                continue

            try:
                collection.upsert(
                    ids=[f"{doc.id}:{idx}"],
                    documents=[content],
                    metadatas=[metadata],
                    embeddings=[vector],
                )
            except Exception:
                LOGGER.exception("Upsert failed chunk %d", idx)


def _query_sweep(
    question: str, max_tokens: int, top_k: int = 5, model: str = "gpt-4o-mini"
):
    """Query from the sweep collection for a given token size."""
    from rag.qa_chain import (
        _get_openai_embedding,
        _call_openai,
        EMBED_MODEL,
        QAResult,
        SourceBlock,
    )
    from prompts.rag_qa import PROMPT
    from utils.logging import log_api_call

    t_total = time.perf_counter()
    coll_name = _collection_name(max_tokens)
    collection = _get_collection(coll_name)

    if collection is None or collection.count() == 0:
        return QAResult(
            question=question,
            answer="[no index]",
            source_blocks=[],
            model=model,
            latency_ms=0,
            input_tokens=0,
            output_tokens=0,
        )

    question_vector, embed_tokens = _get_openai_embedding(question)
    log_api_call(
        model=EMBED_MODEL,
        stage="sweep_query_embed",
        input_tokens=embed_tokens,
        output_tokens=0,
        latency_ms=0,
        cost_usd=0,
        success=True,
    )

    n_results = min(top_k, collection.count())
    raw = collection.query(
        query_embeddings=[question_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    contents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]

    source_blocks: list[SourceBlock] = []
    context_parts: list[str] = []

    for i, content in enumerate(contents):
        meta = metadatas[i] if i < len(metadatas) else {}
        source = meta.get("source", "unknown")
        page = meta.get("page")
        block_order = meta.get("block_order", i)

        source_blocks.append(
            SourceBlock(
                document_id=meta.get("document_id", ""),
                source=source,
                block_order=block_order,
                block_type=meta.get("block_type", "text"),
                content_preview=content[:200],
                page=page,
                cell_index=None,
            )
        )
        context_parts.append(f"[{source}] page {page or block_order}\n{content}")

    context = "\n\n---\n\n".join(context_parts)
    user_content = f"Context:\n{context}\n\nQuestion: {question}"

    raw_answer, input_tokens, output_tokens = _call_openai(model, PROMPT, user_content)
    log_api_call(
        model=model,
        stage="sweep_generate",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=0,
        cost_usd=0,
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


# ---------------------------------------------------------------------------
# DeepEval runner
# ---------------------------------------------------------------------------


def _run_deepeval_for_tokens(max_tokens: int, model: str) -> dict:
    """Run DeepEval for a single token size. Returns serialised EvalReport dict."""
    from eval.evaluator import EvalCase, run_evaluation

    golden_data = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))
    items = golden_data.get("items", [])

    cases = []
    for item in items:
        question = item["question"]
        expected = item["expected_answer"]
        case_id = item.get("id", "unknown")
        tier = item.get("tier", 0)

        LOGGER.info("  [%dtok] querying %s", max_tokens, case_id)
        try:
            result = _query_sweep(question, max_tokens, model=model)
            actual_answer = result.answer
            retrieved_contexts = [
                f"[{sb.source}] page {sb.page if sb.page is not None else sb.block_order}\n{sb.content_preview}"
                for sb in result.source_blocks
            ]
            sources = [sb.source for sb in result.source_blocks]
        except Exception as exc:
            LOGGER.warning("Query failed for %s: %s", case_id, exc)
            actual_answer = f"[ERROR] {exc}"
            retrieved_contexts = []
            sources = []

        cases.append(
            EvalCase(
                question=question,
                expected_answer=expected,
                actual_answer=actual_answer,
                retrieved_contexts=retrieved_contexts,
                sources=sources,
                case_id=case_id,
                tier=tier,
            )
        )

    report = run_evaluation(cases, model=model)
    return asdict(report)


# ---------------------------------------------------------------------------
# Load existing round results
# ---------------------------------------------------------------------------


def _load_existing(token_size: int) -> Optional[dict]:
    """Load a pre-existing round result for this token size, if available."""
    path = _EXISTING_RESULTS.get(token_size)
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Could not load existing result for %d tok: %s", token_size, exc)
        return None


# ---------------------------------------------------------------------------
# Tier breakdown helper
# ---------------------------------------------------------------------------


def _tier_breakdown(report_dict: dict) -> dict:
    """Compute Tier 1 vs Tier 2+ averages from a report dict."""
    golden = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))
    tier_map = {item["id"]: item["tier"] for item in golden["items"]}

    cases = report_dict.get("per_case_results", [])
    tiers: dict[str, list] = {"1": [], "2plus": []}
    for c in cases:
        t = tier_map.get(c.get("case_id", ""), 0)
        if t == 1:
            tiers["1"].append(c)
        elif t >= 2:
            tiers["2plus"].append(c)

    def avg(lst, key):
        return round(sum(x[key] for x in lst) / len(lst), 4) if lst else 0.0

    result = {}
    for label, lst in [("tier_1", tiers["1"]), ("tier_2plus", tiers["2plus"])]:
        result[label] = {
            "n": len(lst),
            "faithfulness": avg(lst, "faithfulness_score"),
            "precision": avg(lst, "context_precision_score"),
            "recall": avg(lst, "context_recall_score"),
            "citation": avg(lst, "citation_score"),
            "overall": avg(lst, "overall_score"),
            "passed": sum(1 for x in lst if x.get("passed")),
        }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point: python -m eval.run_chunking_sweep."""
    parser = argparse.ArgumentParser(
        description="PDF chunking size sweep — DeepEval 4 metrics."
    )
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=_DEFAULT_TOKENS,
        help="Token sizes to run (default: 750 1000)",
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--output", type=Path, default=_OUTPUT_PATH)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip re-running sizes that have existing round results",
    )
    args = parser.parse_args(argv)

    sweep: dict[str, dict] = {}

    # Merge existing results (300/400/500)
    for tok, path in sorted(_EXISTING_RESULTS.items()):
        existing = _load_existing(tok)
        if existing:
            LOGGER.info("Loaded existing result for %d tok from %s", tok, path.name)
            sweep[str(tok)] = {
                "max_tokens": tok,
                "source": "existing",
                "source_file": str(path),
                "report": existing,
                "tier_breakdown": _tier_breakdown(existing),
            }

    # Run new token sizes
    for tok in sorted(args.tokens):
        if args.skip_existing and str(tok) in sweep:
            LOGGER.info("Skipping %d tok (already have result)", tok)
            continue

        LOGGER.info("=== Indexing + evaluating max_tokens=%d ===", tok)
        _index_pdfs_for_tokens(tok)
        report = _run_deepeval_for_tokens(tok, args.model)

        sweep[str(tok)] = {
            "max_tokens": tok,
            "source": "new",
            "report": report,
            "tier_breakdown": _tier_breakdown(report),
        }

        r = report
        print(f"\n=== max_tokens={tok} ===")
        print(f"  Faithfulness:  {r['faithfulness_score']:.4f}")
        print(f"  Precision:     {r['context_precision_score']:.4f}")
        print(f"  Recall:        {r['context_recall_score']:.4f}")
        print(f"  Citation:      {r['citation_score']:.4f}")
        print(f"  Overall:       {r['overall_score']:.4f}")
        print(f"  Passed:        {r['passed_cases']}/{r['total_cases']}")

    # Summary table
    print("\n=== SWEEP SUMMARY ===")
    print(
        f"{'tok':>6}  {'Faith':>6}  {'Prec':>6}  {'Recall':>6}  {'Cit':>6}  {'Overall':>7}  {'Passed':>6}"
    )
    for tok_str in sorted(sweep, key=int):
        entry = sweep[tok_str]
        r = entry["report"]
        print(
            f"{tok_str:>6}  {r['faithfulness_score']:>6.4f}  "
            f"{r['context_precision_score']:>6.4f}  {r['context_recall_score']:>6.4f}  "
            f"{r['citation_score']:>6.4f}  {r['overall_score']:>7.4f}  "
            f"{r['passed_cases']}/{r['total_cases']}"
        )

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "golden_set": str(_GOLDEN_PATH),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "sweep": sweep,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("Sweep results saved to %s", args.output)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()

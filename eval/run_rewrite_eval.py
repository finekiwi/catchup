"""Before/After RAG evaluation: vanilla query vs query rewriting (CU-13).

Runs both pipelines on the combined 31-question golden set (PDF 16 + ipynb 15),
then computes subgroup analysis for rewrite-needed cases and Wilcoxon Signed-Rank Test.

Each case is checkpointed to JSONL immediately after query + eval, so a crashed
run can be resumed without re-running completed cases.

Usage:
    python -m eval.run_rewrite_eval
    python -m eval.run_rewrite_eval --model gpt-4o-mini --output eval/results/rewrite_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)

_GOLDEN_PDF_PATH = Path("eval/golden_set.json")
_GOLDEN_IPYNB_PATH = Path("eval/golden_set_ipynb.json")
_OUTPUT_PATH = Path("eval/results/rewrite_eval.json")
_DEFAULT_MODEL = "gpt-4o-mini"
_REWRITE_MODEL = "gpt-4.1-nano"

# ---------------------------------------------------------------------------
# Manual tagging: cases where query rewriting is expected to improve retrieval.
# Criteria:
#   - Korean technical terms that need English expansion for better embedding match
#   - Abbreviations whose full names appear in the document text
#   - Mixed KO/EN queries against Korean-language documents
# ---------------------------------------------------------------------------
REWRITE_NEEDED_CASES: set[str] = {
    "gs_005",        # "MLP" abbreviation → "MLP (Multilayer Perceptron)" helps match EN+KO docs
    "gs_008",        # EN query against KO git book; "inline commit" → "인라인 커밋 (inline commit, git commit -m)"
    "gs_016",        # KO colloquial compound "커밋로그" → "커밋 로그 (commit log, git log, 기록보기)"; vanilla retrieves commit-intro chunks instead of git log section
    "gs_ipynb_003",  # "RAG" acronym → "RAG (Retrieval-Augmented Generation, 검색 증강 생성)" in KO notebook
    "gs_ipynb_008",  # mixed KO query containing "노드 조회" — KO/EN expansion helpful
    "gs_ipynb_009",  # EN query about KO RAG pipeline; "search_db tool" + "RAG pipeline" expansion helpful
    "gs_ipynb_010",  # pure Korean query "삼성전자는 어떤 기술을 개발하고 있나요?" against KO notebook
}


def _load_golden_items() -> list[dict]:
    """Load and merge PDF + ipynb golden set items."""
    items: list[dict] = []
    for path in (_GOLDEN_PDF_PATH, _GOLDEN_IPYNB_PATH):
        data = json.loads(path.read_text(encoding="utf-8"))
        items.extend(data.get("items", []))
    LOGGER.info("Loaded %d golden items total", len(items))
    return items


# ---------------------------------------------------------------------------
# Case-level checkpoint helpers
# ---------------------------------------------------------------------------


def _case_ckpt_path(output: Path, label: str) -> Path:
    """Return per-case JSONL checkpoint path for a pipeline label."""
    return output.parent / f".ckpt_{output.stem}_{label.lower()}_cases.jsonl"


def _load_case_checkpoint(path: Path) -> dict[str, dict]:
    """Load completed cases from JSONL checkpoint.

    Returns:
        Dict mapping case_id → case_data for all successfully checkpointed cases.
    """
    if not path.exists():
        return {}
    completed: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            completed[d["case_id"]] = d
        except Exception as exc:
            LOGGER.warning("Skipping malformed checkpoint line: %s", exc)
    LOGGER.info("Loaded %d completed cases from %s", len(completed), path)
    return completed


def _append_case_checkpoint(path: Path, case_data: dict) -> None:
    """Append one completed case to JSONL checkpoint (atomic line append)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(case_data, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Pipeline runner — case-by-case query + eval + checkpoint
# ---------------------------------------------------------------------------


def _run_pipeline_with_ckpt(
    query_fn,
    label: str,
    model: str,
    items: list[dict],
    ckpt_path: Path,
) -> tuple[dict, list[float], list[str | None], list[list[str]]]:
    """Run one pipeline (vanilla or rewrite) case-by-case with per-case JSONL checkpoint.

    For each item:
      1. If case_id already in checkpoint → load from checkpoint (skip query + eval).
      2. Otherwise → run query, then evaluate single case, then append to JSONL.

    Args:
        query_fn: Callable(question, top_k, model) → QAResult.
        label: Human-readable pipeline name ("Vanilla" or "Rewrite").
        model: LLM model for answer generation.
        items: Golden set items list.
        ckpt_path: JSONL checkpoint file path.

    Returns:
        (report_dict, per_case_scores, rewritten_queries, retrieved_contexts)
    """
    # Import here to avoid top-level circular import risks
    from eval.evaluator import (  # type: ignore[attr-defined]
        EvalCase,
        _evaluate_single_case,
        _resolve_judge_model,
        _build_citation_metric,
    )
    from deepeval.metrics import (
        FaithfulnessMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
    )

    completed = _load_case_checkpoint(ckpt_path)
    remaining = sum(1 for item in items if item.get("id", "unknown") not in completed)
    LOGGER.info("=== Pipeline: %s — %d/%d to run ===", label, remaining, len(items))

    judge_model = _resolve_judge_model(model)

    # Build metric instances once and reuse across cases
    faithfulness_metric = FaithfulnessMetric(threshold=0.5, model=judge_model, include_reason=True)
    context_precision_metric = ContextualPrecisionMetric(threshold=0.5, model=judge_model, include_reason=True)
    context_recall_metric = ContextualRecallMetric(threshold=0.5, model=judge_model, include_reason=True)
    citation_metric = _build_citation_metric(judge_model)

    all_case_data: list[dict] = []

    for item in items:
        case_id: str = item.get("id", "unknown")
        question: str = item["question"]
        expected: str = item["expected_answer"]
        tier: int = item.get("tier", 0)

        # Resume: use checkpoint data if available
        if case_id in completed:
            LOGGER.info("  [%s] %s — loaded from checkpoint", label, case_id)
            all_case_data.append(completed[case_id])
            continue

        # --- Query phase ---
        LOGGER.info("  [%s] %s — querying", label, case_id)
        try:
            result = query_fn(question, top_k=5, model=model)
            actual_answer = result.answer
            retrieved_contexts = [
                f"[{sb.source}] page {sb.page if sb.page is not None else sb.block_order}\n{sb.content_preview}"
                for sb in result.source_blocks
            ]
            sources = [sb.source for sb in result.source_blocks]
            rewritten_query: str | None = getattr(result, "rewritten_query", None)
        except Exception as exc:
            LOGGER.warning("%s query failed for %s: %s", label, case_id, exc)
            actual_answer = f"[ERROR] {exc}"
            retrieved_contexts = []
            sources = []
            rewritten_query = None

        # --- Eval phase ---
        LOGGER.info("  [%s] %s — evaluating", label, case_id)
        eval_case = EvalCase(
            question=question,
            expected_answer=expected,
            actual_answer=actual_answer,
            retrieved_contexts=retrieved_contexts,
            sources=sources,
            case_id=case_id,
            tier=tier,
        )
        case_result = _evaluate_single_case(
            eval_case,
            faithfulness_metric,
            context_precision_metric,
            context_recall_metric,
            citation_metric,
        )

        case_data = {
            "case_id": case_id,
            "question": question,
            "actual_answer": actual_answer,
            "rewritten_query": rewritten_query,
            "retrieved_contexts": retrieved_contexts,
            "faithfulness_score": case_result.faithfulness_score,
            "context_precision_score": case_result.context_precision_score,
            "context_recall_score": case_result.context_recall_score,
            "citation_score": case_result.citation_score,
            "overall_score": case_result.overall_score,
            "passed": case_result.passed,
            "tier": tier,
            "error": case_result.error,
        }

        _append_case_checkpoint(ckpt_path, case_data)
        all_case_data.append(case_data)
        LOGGER.info(
            "  [%s] %s — done: overall=%.4f (faith=%.4f, prec=%.4f, rec=%.4f, cit=%.4f)",
            label, case_id, case_result.overall_score,
            case_result.faithfulness_score, case_result.context_precision_score,
            case_result.context_recall_score, case_result.citation_score,
        )

    # Reconstruct aggregate report from all_case_data
    n = len(all_case_data)
    agg_faithfulness = sum(d["faithfulness_score"] for d in all_case_data) / n
    agg_precision = sum(d["context_precision_score"] for d in all_case_data) / n
    agg_recall = sum(d["context_recall_score"] for d in all_case_data) / n
    agg_citation = sum(d["citation_score"] for d in all_case_data) / n
    agg_overall = sum(d["overall_score"] for d in all_case_data) / n
    passed = sum(1 for d in all_case_data if d.get("passed", False))

    report: dict = {
        "overall_score": round(agg_overall, 4),
        "faithfulness_score": round(agg_faithfulness, 4),
        "context_precision_score": round(agg_precision, 4),
        "context_recall_score": round(agg_recall, 4),
        "citation_score": round(agg_citation, 4),
        "total_cases": n,
        "passed_cases": passed,
        "model": model,
        "judge_model": judge_model,
        "per_case_results": all_case_data,
    }

    print(f"\n=== {label} ===")
    print(f"  Faithfulness:       {report['faithfulness_score']:.4f}")
    print(f"  Context Precision:  {report['context_precision_score']:.4f}")
    print(f"  Context Recall:     {report['context_recall_score']:.4f}")
    print(f"  Citation Accuracy:  {report['citation_score']:.4f}")
    print(f"  Overall:            {report['overall_score']:.4f}")
    print(f"  Passed:             {report['passed_cases']}/{report['total_cases']}")

    per_case_scores = [d["overall_score"] for d in all_case_data]
    rewritten_queries: list[str | None] = [d["rewritten_query"] for d in all_case_data]
    retrieved_contexts: list[list[str]] = [d["retrieved_contexts"] for d in all_case_data]

    return report, per_case_scores, rewritten_queries, retrieved_contexts


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def _binary_comparison(
    items: list[dict],
    vanilla_scores: list[float],
    rewrite_scores: list[float],
    rewrite_queries: list[str | None],
    vanilla_contexts: list[list[str]],
    rewrite_contexts: list[list[str]],
) -> list[dict]:
    """Build per-case binary comparison list with retrieval context diff."""
    _PASS_THRESHOLD = 0.5
    result = []
    for i, item in enumerate(items):
        case_id = item.get("id", f"case_{i}")
        result.append({
            "case_id": case_id,
            "question": item["question"],
            "rewritten_query": rewrite_queries[i],
            "rewrite_needed": case_id in REWRITE_NEEDED_CASES,
            "vanilla_overall": round(vanilla_scores[i], 4),
            "rewrite_overall": round(rewrite_scores[i], 4),
            "vanilla_correct": vanilla_scores[i] >= _PASS_THRESHOLD,
            "rewrite_correct": rewrite_scores[i] >= _PASS_THRESHOLD,
            "vanilla_retrieved": vanilla_contexts[i],
            "rewrite_retrieved": rewrite_contexts[i],
        })
    return result


def _subgroup_analysis(per_case: list[dict]) -> dict:
    """Compute subgroup metrics for rewrite_needed vs not_needed cases."""
    needed = [c for c in per_case if c["rewrite_needed"]]
    not_needed = [c for c in per_case if not c["rewrite_needed"]]

    def _stats(subset: list[dict]) -> dict:
        if not subset:
            return {"count": 0, "vanilla_overall": 0.0, "rewrite_overall": 0.0, "delta": 0.0}
        v_avg = sum(c["vanilla_overall"] for c in subset) / len(subset)
        r_avg = sum(c["rewrite_overall"] for c in subset) / len(subset)
        return {
            "count": len(subset),
            "vanilla_overall": round(v_avg, 4),
            "rewrite_overall": round(r_avg, 4),
            "delta": round(r_avg - v_avg, 4),
        }

    return {
        "rewrite_needed": _stats(needed),
        "rewrite_not_needed": _stats(not_needed),
    }


def _wilcoxon_test(vanilla_scores: list[float], rewrite_scores: list[float]) -> dict:
    """Run Wilcoxon Signed-Rank Test on paired score lists."""
    try:
        from scipy.stats import wilcoxon

        n = len(vanilla_scores)
        differences = [r - v for v, r in zip(vanilla_scores, rewrite_scores)]
        if all(d == 0 for d in differences):
            return {"test": "wilcoxon", "n": n, "statistic": None, "p_value": None,
                    "significant": False, "note": "all differences are zero"}
        stat, p_value = wilcoxon(rewrite_scores, vanilla_scores)
        return {
            "test": "wilcoxon",
            "n": n,
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 6),
            "significant": bool(p_value < 0.05),
        }
    except Exception as exc:
        LOGGER.warning("Wilcoxon test failed: %s", exc)
        return {"test": "wilcoxon", "error": str(exc)}


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------


def run_rewrite_eval(model: str, output: Path = Path("eval/results/rewrite_eval.json")) -> dict:
    """Run vanilla vs rewrite evaluation on the combined 31Q golden set.

    Each case is checkpointed to JSONL immediately after query + eval, so a
    mid-run crash can be resumed from the last completed case.

    Args:
        model: LLM model for answer generation (same for both pipelines).
        output: Final output path (used to derive checkpoint paths).

    Returns:
        Full result dict with pipelines, binary_comparison, subgroup_analysis,
        and statistical_test sections.
    """
    from functools import partial
    from rag.qa_chain import query_chunked

    items = _load_golden_items()

    ckpt_vanilla = _case_ckpt_path(output, "vanilla")
    ckpt_rewrite = _case_ckpt_path(output, "rewrite")

    vanilla_report, vanilla_scores, _, vanilla_contexts = _run_pipeline_with_ckpt(
        partial(query_chunked, rewrite=False),
        label="Vanilla",
        model=model,
        items=items,
        ckpt_path=ckpt_vanilla,
    )

    rewrite_report, rewrite_scores, rewrite_queries, rewrite_contexts = _run_pipeline_with_ckpt(
        partial(query_chunked, rewrite=True),
        label="Rewrite",
        model=model,
        items=items,
        ckpt_path=ckpt_rewrite,
    )

    per_case = _binary_comparison(
        items, vanilla_scores, rewrite_scores, rewrite_queries, vanilla_contexts, rewrite_contexts
    )
    subgroup = _subgroup_analysis(per_case)

    # Wilcoxon tests: all 31 + rewrite_needed subset only
    needed_indices = [i for i, c in enumerate(per_case) if c["rewrite_needed"]]
    vanilla_needed = [vanilla_scores[i] for i in needed_indices]
    rewrite_needed_scores = [rewrite_scores[i] for i in needed_indices]

    stat_all = _wilcoxon_test(vanilla_scores, rewrite_scores)
    stat_needed = (
        _wilcoxon_test(vanilla_needed, rewrite_needed_scores)
        if needed_indices else {"note": "no rewrite_needed cases"}
    )

    vanilla_correct = sum(1 for c in per_case if c["vanilla_correct"])
    rewrite_correct = sum(1 for c in per_case if c["rewrite_correct"])

    print(f"\n=== Binary Comparison ===")
    print(f"  Total: {len(per_case)}")
    print(f"  Vanilla correct:  {vanilla_correct}/{len(per_case)}")
    print(f"  Rewrite correct:  {rewrite_correct}/{len(per_case)}")
    print(f"\n=== Subgroup Analysis ===")
    for grp, stats in subgroup.items():
        print(f"  {grp}: n={stats['count']}, vanilla={stats['vanilla_overall']:.4f}, "
              f"rewrite={stats['rewrite_overall']:.4f}, delta={stats['delta']:+.4f}")
    print(f"\n=== Statistical Test (all) ===")
    print(f"  p={stat_all.get('p_value')}, significant={stat_all.get('significant')}")
    print(f"=== Statistical Test (rewrite_needed) ===")
    print(f"  p={stat_needed.get('p_value')}, significant={stat_needed.get('significant')}")

    return {
        "golden_set": f"{_GOLDEN_PDF_PATH} + {_GOLDEN_IPYNB_PATH}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "answer_model": model,
        "rewrite_model": _REWRITE_MODEL,
        "pipelines": {
            "vanilla": vanilla_report,
            "rewrite": rewrite_report,
        },
        "binary_comparison": {
            "total": len(per_case),
            "vanilla_correct": vanilla_correct,
            "rewrite_correct": rewrite_correct,
            "per_case": per_case,
        },
        "subgroup_analysis": subgroup,
        "statistical_test": {
            "all_cases": stat_all,
            "rewrite_needed_only": stat_needed,
        },
    }


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point: python -m eval.run_rewrite_eval."""
    parser = argparse.ArgumentParser(
        description="Run Before/After RAG evaluation: vanilla vs query rewriting (CU-13)."
    )
    parser.add_argument("--model", type=str, default=_DEFAULT_MODEL, help="Answer LLM model")
    parser.add_argument(
        "--output", type=Path, default=_OUTPUT_PATH,
        help=f"Output JSON path (default: {_OUTPUT_PATH})",
    )
    args = parser.parse_args(argv)

    for path in (_GOLDEN_PDF_PATH, _GOLDEN_IPYNB_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Golden set not found: {path}")

    results = run_rewrite_eval(args.model, output=args.output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Results saved to %s", args.output)
    print(f"\nRewrite eval complete. Results saved to: {args.output}")


if __name__ == "__main__":
    main()
